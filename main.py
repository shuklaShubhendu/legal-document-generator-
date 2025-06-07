import streamlit as st
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any
import openai
from openai import OpenAI
import PyPDF2
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import json
import re
from datetime import datetime

# Initialize OpenAI client with API key
@st.cache_resource
def get_openai_client(api_key):
    if not api_key:
        return None
    return OpenAI(api_key=api_key)

class LegalDocumentGenerator:
    def __init__(self, api_key=None):
        self.api_key = api_key
        self.client = None
        self.embeddings = None
        self.vectorstore = None
        self.knowledge_base_path = "knowledge_base"
        self.vectorstore_path = "vectorstore.pkl"
        
        # Initialize clients only if API key is provided
        if api_key:
            self._initialize_clients()
    
    def _initialize_clients(self):
        """Initialize OpenAI client and embeddings"""
        try:
            self.client = get_openai_client(self.api_key)
            self.embeddings = OpenAIEmbeddings(openai_api_key=self.api_key)
        except Exception as e:
            st.error(f"Error initializing OpenAI clients: {str(e)}")
            self.client = None
            self.embeddings = None
    
    def set_api_key(self, api_key):
        """Set API key and initialize clients"""
        self.api_key = api_key
        if api_key:
            self._initialize_clients()
            return True
        return False
        
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            st.error(f"Error reading PDF {pdf_path}: {str(e)}")
        return text
    
    def load_and_process_pdfs(self) -> List[Document]:
        """Load and process all PDFs in the knowledge base folder"""
        documents = []
        pdf_folder = Path(self.knowledge_base_path)
        
        if not pdf_folder.exists():
            st.error(f"Knowledge base folder '{self.knowledge_base_path}' not found!")
            return documents
            
        pdf_files = list(pdf_folder.glob("*.pdf"))
        
        if not pdf_files:
            st.error("No PDF files found in the knowledge base folder!")
            return documents
            
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, pdf_file in enumerate(pdf_files):
            status_text.text(f"Processing {pdf_file.name}...")
            text = self.extract_text_from_pdf(str(pdf_file))
            
            if text.strip():
                # Split text into chunks
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    length_function=len
                )
                chunks = text_splitter.split_text(text)
                
                for j, chunk in enumerate(chunks):
                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": pdf_file.name,
                            "chunk_id": j,
                            "total_chunks": len(chunks)
                        }
                    )
                    documents.append(doc)
            
            progress_bar.progress((i + 1) / len(pdf_files))
        
        status_text.text("Processing complete!")
        return documents
    
    def create_vectorstore(self, documents: List[Document]):
        """Create and save vectorstore from documents"""
        if not self.embeddings:
            st.error("OpenAI embeddings not initialized. Please provide a valid API key.")
            return False
            
        if documents:
            try:
                self.vectorstore = FAISS.from_documents(documents, self.embeddings)
                # Save vectorstore
                with open(self.vectorstore_path, 'wb') as f:
                    pickle.dump(self.vectorstore, f)
                st.success(f"Vectorstore created with {len(documents)} document chunks!")
                return True
            except Exception as e:
                st.error(f"Error creating vectorstore: {str(e)}")
                return False
        else:
            st.error("No documents to create vectorstore!")
            return False
    
    def load_vectorstore(self):
        """Load existing vectorstore"""
        try:
            with open(self.vectorstore_path, 'rb') as f:
                self.vectorstore = pickle.load(f)
            st.success("Existing knowledge base loaded successfully!")
            return True
        except FileNotFoundError:
            if self.embeddings:
                st.info("No existing vectorstore found. Processing PDFs...")
                documents = self.load_and_process_pdfs()
                return self.create_vectorstore(documents)
            else:
                st.warning("No vectorstore found and no API key provided. Please provide OpenAI API key to process PDFs.")
                return False
    
    def search_knowledge_base(self, query: str, k: int = 5) -> List[str]:
        """Search the knowledge base for relevant information"""
        if not self.vectorstore:
            return []
        
        results = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]
    
    def analyze_legal_case(self, case_description: str) -> Dict[str, Any]:
        """Analyze the legal case and determine document type and required information"""
        
        if not self.client:
            st.error("OpenAI client not initialized. Please provide a valid API key.")
            return {
                "document_type": "General Legal Document",
                "legal_provisions": ["API key required for analysis"],
                "required_info": ["Please provide OpenAI API key"],
                "follow_up_questions": ["Please configure your OpenAI API key first"]
            }
        
        relevant_docs = self.search_knowledge_base(case_description, k=3)
        context = "\n\n".join(relevant_docs)
        
        analysis_prompt = f"""
        You are an expert Indian legal AI assistant. Analyze the following case description and provide:
        1. Document type needed (e.g., Civil Suit, Criminal Complaint, Writ Petition, etc.)
        2. Relevant legal provisions/sections
        3. Required information that needs to be gathered
        4. Follow-up questions to ask the lawyer
        
        Case Description: {case_description}
        
        Relevant Legal Knowledge:
        {context}
        
        Provide your response in JSON format with keys: document_type, legal_provisions, required_info, follow_up_questions
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.3
            )
            
            # Try to parse JSON response
            content = response.choices[0].message.content
            # Extract JSON from response if it's wrapped in markdown
            json_match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
            if json_match:
                content = json_match.group(1)
            
            return json.loads(content)
        except Exception as e:
            st.error(f"Error analyzing case: {str(e)}")
            return {
                "document_type": "General Legal Document",
                "legal_provisions": ["To be determined"],
                "required_info": ["Basic case details"],
                "follow_up_questions": ["Please provide more details about your case"]
            }
    
    def generate_legal_document(self, case_info: Dict[str, Any], document_type: str) -> str:
        """Generate legal document in LaTeX format"""
        
        if not self.client:
            return "Error: OpenAI client not initialized. Please provide a valid API key."
        
        # Search for relevant templates and precedents
        search_query = f"{document_type} template format Indian law"
        relevant_docs = self.search_knowledge_base(search_query, k=5)
        context = "\n\n".join(relevant_docs)
        
        generation_prompt = f"""
        You are an expert Indian legal document drafter. Generate a complete {document_type} in LaTeX format.
        
        Case Information:
        {json.dumps(case_info, indent=2, default=str)}
        
        Relevant Legal Templates and Precedents:
        {context}
        
        Requirements:
        1. Use proper LaTeX document structure
        2. Include all necessary legal sections and clauses
        3. Follow Indian legal formatting standards
        4. Include proper citations and references
        5. Use appropriate legal language and terminology
        6. Include placeholders for missing information in [PLACEHOLDER] format
        
        Generate a complete, professional legal document in LaTeX format.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": generation_prompt}],
                temperature=0.2,
                max_tokens=4000
            )
            
            return response.choices[0].message.content
        except Exception as e:
            st.error(f"Error generating document: {str(e)}")
            return "Error generating document. Please try again."

def main():
    st.set_page_config(
        page_title="Legal Document Generator",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ Indian Legal Document Generator")
    st.markdown("*AI-Powered RAG System for Legal Document Drafting*")
    
    # Initialize the generator without API key initially
    if 'generator' not in st.session_state:
        st.session_state.generator = LegalDocumentGenerator()
    
    # Main interface
    st.header("🔑 API Configuration")
    
    # Get API key from user
    api_key = st.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        help="Your OpenAI API key is required to process documents and generate legal content"
    )
    
    if api_key:
        # Set API key and initialize clients
        if st.session_state.generator.set_api_key(api_key):
            st.success("✅ API key configured successfully!")
            
            # Load knowledge base only after API key is set
            if not st.session_state.generator.vectorstore:
                with st.spinner("Loading knowledge base..."):
                    st.session_state.generator.load_vectorstore()
        else:
            st.error("❌ Failed to configure API key")
    else:
        st.warning("⚠️ Please enter your OpenAI API key to continue")
        st.info("You can get your API key from: https://platform.openai.com/api-keys")
        st.stop()  # Stop execution here if no API key
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        
        st.markdown("---")
        st.header("📚 Knowledge Base")
        
        # Knowledge base status
        if st.session_state.generator.vectorstore:
            st.success("✅ Knowledge Base Loaded")
            
            # Refresh knowledge base button
            if st.button("🔄 Refresh Knowledge Base"):
                with st.spinner("Reprocessing PDFs..."):
                    documents = st.session_state.generator.load_and_process_pdfs()
                    st.session_state.generator.create_vectorstore(documents)
        else:
            st.error("❌ Knowledge Base Not Available")
            st.info("Knowledge base will load after API key is configured")
    
    # Main interface
    st.header("📝 Case Description")
    case_description = st.text_area(
        "Describe your legal case:",
        placeholder="e.g., Client wants to file for recovery of ₹4 lakh. We had sent a legal notice for cheque bounce but no response.",
        height=100
    )
    
    if case_description and st.button("🔍 Analyze Case", type="primary"):
        with st.spinner("Analyzing case..."):
            analysis = st.session_state.generator.analyze_legal_case(case_description)
            st.session_state.analysis = analysis
    
    # Display analysis results
    if 'analysis' in st.session_state:
        st.header("📊 Case Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Document Type")
            st.info(st.session_state.analysis.get('document_type', 'Not determined'))
            
            st.subheader("Legal Provisions")
            provisions = st.session_state.analysis.get('legal_provisions', [])
            for provision in provisions:
                st.write(f"• {provision}")
        
        with col2:
            st.subheader("Required Information")
            required_info = st.session_state.analysis.get('required_info', [])
            for info in required_info:
                st.write(f"• {info}")
        
        # Follow-up questions
        st.subheader("🤔 Follow-up Questions")
        questions = st.session_state.analysis.get('follow_up_questions', [])
        
        # Create a form for collecting additional information
        with st.form("additional_info_form"):
            st.markdown("Please provide the following information:")
            
            additional_info = {}
            for i, question in enumerate(questions):
                additional_info[f"question_{i}"] = st.text_input(question)
            
            # Additional fields based on common legal requirements
            st.markdown("**Additional Details:**")
            additional_info['client_name'] = st.text_input("Client Name")
            additional_info['opposing_party'] = st.text_input("Opposing Party Name")
            additional_info['date_of_incident'] = st.date_input("Date of Incident")
            additional_info['amount_claimed'] = st.text_input("Amount Claimed (if applicable)")
            additional_info['additional_details'] = st.text_area("Any Additional Details")
            
            if st.form_submit_button("📄 Generate Document", type="primary"):
                if any(additional_info.values()):
                    # Combine original case description with additional info
                    case_info = {
                        'original_description': case_description,
                        'document_type': st.session_state.analysis.get('document_type'),
                        'additional_info': additional_info,
                        'analysis': st.session_state.analysis
                    }
                    
                    with st.spinner("Generating legal document..."):
                        latex_document = st.session_state.generator.generate_legal_document(
                            case_info, 
                            st.session_state.analysis.get('document_type', 'Legal Document')
                        )
                        st.session_state.generated_document = latex_document
                else:
                    st.warning("Please provide at least some additional information to generate the document.")
    
    # Display generated document
    if 'generated_document' in st.session_state:
        st.header("📄 Generated Legal Document")
        
        # Document preview
        with st.expander("📖 Document Preview", expanded=True):
            st.code(st.session_state.generated_document, language="latex")
        
        # Download options
        st.subheader("💾 Download Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.download_button(
                label="📄 Download LaTeX",
                data=st.session_state.generated_document,
                file_name=f"legal_document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex",
                mime="text/plain"
            )
        
        with col2:
            # Note about PDF conversion
            st.info("💡 To convert LaTeX to PDF, use online tools like Overleaf or local LaTeX compiler")
        
        with col3:
            # Note about DOCX conversion
            st.info("💡 Use pandoc or online converters to convert LaTeX to DOCX format")
        
        # Regenerate option
        if st.button("🔄 Regenerate Document"):
            with st.spinner("Regenerating document..."):
                case_info = {
                    'original_description': case_description,
                    'document_type': st.session_state.analysis.get('document_type'),
                    'analysis': st.session_state.analysis
                }
                latex_document = st.session_state.generator.generate_legal_document(
                    case_info, 
                    st.session_state.analysis.get('document_type', 'Legal Document')
                )
                st.session_state.generated_document = latex_document
                st.rerun()

if __name__ == "__main__":
    main()