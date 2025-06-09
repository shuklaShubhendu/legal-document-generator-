import streamlit as st
import os
import pickle
from pathlib import Path
from typing import List, Dict, Any
import openai
from openai import OpenAI
# pypdf is the successor to PyPDF2 and is actively maintained
import pypdf
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
        # CORRECTED: FAISS saves to a directory, not a single file.
        self.vectorstore_path = "faiss_index"
        
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
        """Extract text from PDF file using the more modern pypdf library"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
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
            
        progress_bar = st.progress(0, text="Initializing PDF processing...")
        status_text = st.empty()
        
        for i, pdf_file in enumerate(pdf_files):
            status_text.text(f"Processing {pdf_file.name}...")
            text = self.extract_text_from_pdf(str(pdf_file))
            
            if text.strip():
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
            
            progress_bar.progress((i + 1) / len(pdf_files), text=f"Processed {i+1}/{len(pdf_files)} files.")
        
        status_text.text("Processing complete!")
        return documents
    
    def create_vectorstore(self, documents: List[Document]):
        """Create and save vectorstore from documents using FAISS's native method"""
        if not self.embeddings:
            st.error("OpenAI embeddings not initialized. Please provide a valid API key.")
            return False
            
        if documents:
            try:
                self.vectorstore = FAISS.from_documents(documents, self.embeddings)
                # CORRECTED: Use the robust `save_local` method instead of pickle
                self.vectorstore.save_local(self.vectorstore_path)
                st.success(f"Vectorstore created with {len(documents)} document chunks!")
                return True
            except Exception as e:
                st.error(f"Error creating vectorstore: {str(e)}")
                return False
        else:
            st.error("No documents to create vectorstore!")
            return False
    
    def load_vectorstore(self):
        """Load existing vectorstore or create a new one using FAISS's native method"""
        vectorstore_dir = Path(self.vectorstore_path)
        
        # CORRECTED: Check if the directory exists and has files in it.
        if vectorstore_dir.exists() and any(vectorstore_dir.iterdir()):
            if not self.embeddings:
                st.warning("Found a knowledge base, but an API key is needed to load it. Please enter your key.")
                return False
            
            try:
                # CORRECTED: Use `load_local` and pass the embeddings object.
                self.vectorstore = FAISS.load_local(
                    self.vectorstore_path, 
                    self.embeddings,
                    allow_dangerous_deserialization=True # This is required by recent LangChain versions
                )
                st.success("Existing knowledge base loaded successfully!")
                return True
            except Exception as e:
                st.warning(f"Could not load existing vectorstore ({e}). This might be due to a version mismatch. Re-creating...")
        
        # This block runs if the directory doesn't exist OR if loading failed.
        if self.embeddings:
            st.info("No existing knowledge base found (or it failed to load). Building a new one from PDFs...")
            documents = self.load_and_process_pdfs()
            return self.create_vectorstore(documents)
        else:
            st.warning("No knowledge base found and no API key provided. Please provide an OpenAI API key to process PDFs.")
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
            return {}
        
        relevant_docs = self.search_knowledge_base(case_description, k=3)
        context = "\n\n".join(relevant_docs)
        
        analysis_prompt = f"""
        You are an expert Indian legal AI assistant. Analyze the following case description and provide a structured analysis.

        Case Description: {case_description}
        
        Relevant Legal Knowledge from internal documents:
        {context}
        
        Provide your response in a clean JSON format with the following keys:
        - "document_type": Suggest a specific document type (e.g., "Civil Suit for Recovery under Order XXXVII", "Criminal Complaint under Section 138 of the Negotiable Instruments Act").
        - "legal_provisions": A list of relevant legal sections and acts (e.g., ["Section 138 of The Negotiable Instruments Act, 1881", "Section 420 of the Indian Penal Code, 1860"]).
        - "required_info": A list of key information that must be gathered from the client (e.g., "Full name and address of the complainant", "Details of the bounced cheque (cheque number, date, amount)").
        - "follow_up_questions": A list of specific, numbered questions to ask the lawyer/client to clarify facts and strengthen the case.
        """
        
        try:
            response = self.client.chat.completions.create(
                # CORRECTED: Using a modern, standard model name
                model="gpt-4o",
                messages=[{"role": "user", "content": analysis_prompt}],
                temperature=0.2,
                response_format={"type": "json_object"} # Use JSON mode for guaranteed JSON output
            )
            
            content = response.choices[0].message.content
            return json.loads(content)

        except Exception as e:
            st.error(f"Error analyzing case: {str(e)}")
            # Fallback in case of API error
            return {
                "document_type": "General Legal Document",
                "legal_provisions": ["Analysis failed, please review case details."],
                "required_info": ["Basic case details"],
                "follow_up_questions": ["Could you provide more specific details about your case?"]
            }
    
    def generate_legal_document(self, case_info: Dict[str, Any], document_type: str) -> str:
        """Generate legal document in LaTeX format using a robust prompt."""
        if not self.client:
            return "Error: OpenAI client not initialized. Please provide a valid API key."
        
        search_query = f"Indian law template and format for {document_type}"
        relevant_docs = self.search_knowledge_base(search_query, k=5)
        context = "\n\n".join(relevant_docs)
        
        generation_prompt = f"""
        You are an expert Indian legal document drafter and a specialist in LaTeX formatting. Your task is to generate a complete, professional, and syntactically correct LaTeX document for a '{document_type}'.

        **Case Information:**
        ```json
        {json.dumps(case_info, indent=2, default=str)}
        ```

        **Relevant Legal Templates and Context from Knowledge Base:**
        ```
        {context}
        ```

        **LaTeX Generation Rules (Follow these strictly):**

        1.  **Complete Document:** The output MUST be a full LaTeX document. It must start with `\\documentclass{{article}}` and end with `\\end{{document}}`.
        2.  **Required Packages:** You MUST include the following packages in the preamble:
            * `\\usepackage[a4paper, margin=1in]{{geometry}}`
            * `\\usepackage{{amsmath}}`
            * `\\usepackage{{amssymb}}`
            * `\\usepackage{{hyperref}}`
            * `\\usepackage{{titling}}`
            * `\\linespread{{1.5}}`

        3.  **Document Structure (Anatomy):** The document MUST follow this structure in this precise order:
            a.  Preamble (documentclass, usepackage).
            b.  Court Details (e.g., `IN THE COURT OF...`).
            c.  Cause Title (Plaintiff/Petitioner vs. Defendant/Respondent). Use a `tabular` environment for clean alignment.
            d.  Subject/Prayer of the document.
            e.  The main body of the document, with paragraphs clearly numbered using an `enumerate` environment.
            f.  Prayer/Relief Sought section.
            g.  Verification clause.
            h.  Signature blocks for the party and advocate.
            i.  Date and Place.

        4.  **CRITICAL - Escape Special Characters:** Before inserting any text from the 'Case Information' into the LaTeX code, you MUST escape all special LaTeX characters. This includes, but is not limited to: `&` -> `\\&`, `%` -> `\\%`, `$` -> `\\$`, `#` -> `\\#`, `_` -> `\\_`.

        5.  **Placeholders:** Use clear, bracketed placeholders like `[INSERT DETAILS OF CHEQUE BOUNCE]` or `[ADVOCATE'S SIGNATURE]` for any information that is missing from the case details.

        6.  **Clarity and Professionalism:** Use precise legal terminology appropriate for Indian law. The final document should be ready for compilation without errors.

        Now, generate the complete LaTeX code based on these strict instructions. Only output the raw LaTeX code, without any explanatory text or markdown code fences.
        """
        
        try:
            response = self.client.chat.completions.create(
                # CORRECTED: Using a modern, standard model name
                model="gpt-4.1",
                messages=[{"role": "user", "content": generation_prompt}],
                temperature=0.2,
                max_tokens=4000
            )
            
            generated_text = response.choices[0].message.content
            
            # Clean up the response to ensure it's just LaTeX
            # Remove markdown code block fences if they exist
            latex_match = re.search(r'```(?:latex)?\n(.*?)\n```', generated_text, re.DOTALL)
            if latex_match:
                return latex_match.group(1).strip()
            return generated_text.strip()

        except Exception as e:
            st.error(f"Error generating document: {str(e)}")
            return f"Error generating document. Please check the logs.\n\nDetails: {str(e)}"

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
    
    # API Key Configuration
    st.header("🔑 API Configuration")
    api_key = st.text_input(
        "Enter your OpenAI API Key:",
        type="password",
        help="Your OpenAI API key is required to process documents and generate legal content"
    )
    
    if api_key:
        if 'api_key_configured' not in st.session_state or not st.session_state.api_key_configured:
            if st.session_state.generator.set_api_key(api_key):
                st.session_state.api_key_configured = True
                st.success("✅ API key configured successfully!")
                # Automatically load knowledge base after key is set
                with st.spinner("Loading knowledge base... This may take a moment."):
                    st.session_state.generator.load_vectorstore()
                st.rerun() # Rerun to update the UI state
            else:
                st.error("❌ Failed to configure API key")
    else:
        st.warning("⚠️ Please enter your OpenAI API key to continue")
        st.info("You can get your API key from: https://platform.openai.com/api-keys")
        st.stop()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("🔧 Configuration")
        st.markdown("---")
        st.header("📚 Knowledge Base")
        
        if st.session_state.generator.vectorstore:
            st.success("✅ Knowledge Base Loaded")
            if st.button("🔄 Refresh Knowledge Base"):
                with st.spinner("Reprocessing PDFs..."):
                    documents = st.session_state.generator.load_and_process_pdfs()
                    st.session_state.generator.create_vectorstore(documents)
                st.rerun()
        else:
            st.error("❌ Knowledge Base Not Available")
            st.info("Knowledge base will load after API key is configured.")
    
    # Stop if API key is not fully configured
    if not st.session_state.get('api_key_configured'):
        st.stop()

    # Main interface
    st.header("📝 Case Description")
    case_description = st.text_area(
        "Describe your legal case:",
        placeholder="e.g., Client wants to file for recovery of ₹4 lakh. We had sent a legal notice for cheque bounce but no response.",
        height=100,
        key="case_description"
    )
    
    if case_description and st.button("🔍 Analyze Case", type="primary"):
        with st.spinner("Analyzing case..."):
            analysis = st.session_state.generator.analyze_legal_case(case_description)
            st.session_state.analysis = analysis
            # Clear previous document when a new analysis is run
            if 'generated_document' in st.session_state:
                del st.session_state['generated_document']
    
    if 'analysis' in st.session_state and st.session_state.analysis:
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
        
        st.subheader("🤔 Follow-up Questions")
        questions = st.session_state.analysis.get('follow_up_questions', [])
        
        with st.form("additional_info_form"):
            st.markdown("Please provide the following information:")
            
            additional_info = {}
            for i, question in enumerate(questions):
                additional_info[question] = st.text_input(question, key=f"q_{i}")
            
            st.markdown("**Standard Details:**")
            additional_info['client_name'] = st.text_input("Client Name", key="client_name")
            additional_info['opposing_party'] = st.text_input("Opposing Party Name", key="opposing_party")
            additional_info['date_of_incident'] = st.date_input("Date of Incident", key="date_incident")
            additional_info['amount_claimed'] = st.text_input("Amount Claimed (if applicable)", key="amount")
            additional_info['additional_details'] = st.text_area("Any Additional Details", key="add_details")
            
            submitted = st.form_submit_button("📄 Generate Document", type="primary")

            if submitted:
                # Filter out empty responses
                provided_info = {k: v for k, v in additional_info.items() if v}

                if not provided_info:
                    st.warning("Please provide some additional information to generate the document.")
                else:
                    case_info = {
                        'original_description': st.session_state.case_description,
                        'document_type': st.session_state.analysis.get('document_type'),
                        'additional_info_provided': provided_info,
                        'case_analysis_summary': st.session_state.analysis
                    }
                    
                    with st.spinner("Generating legal document... This can take up to a minute."):
                        latex_document = st.session_state.generator.generate_legal_document(
                            case_info, 
                            st.session_state.analysis.get('document_type', 'Legal Document')
                        )
                        st.session_state.generated_document = latex_document
                        st.rerun()
    
    if 'generated_document' in st.session_state:
        st.header("📄 Generated Legal Document")
        
        with st.expander("📖 LaTeX Code Preview", expanded=True):
            st.code(st.session_state.generated_document, language="latex")
        
        st.subheader("💾 Download Options")
        
        st.download_button(
            label="📄 Download .tex File",
            data=st.session_state.generated_document.encode('utf-8'), # Encode for safety
            file_name=f"legal_document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tex",
            mime="text/x-tex"
        )
        
        st.info("💡 To convert LaTeX to PDF, use an online editor like Overleaf or a local TeX distribution (MiKTeX, TeX Live).")
        
if __name__ == "__main__":
    main()