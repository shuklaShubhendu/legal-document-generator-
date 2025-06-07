#!/usr/bin/env python3
"""
Setup script for Legal Document Generator
This script helps set up the project structure and dependencies
"""

import os
import sys
import subprocess
from pathlib import Path

def create_directory_structure():
    """Create necessary directories"""
    directories = [
        "knowledge_base",
        "generated_documents",
        "temp"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created directory: {directory}")

def install_requirements():
    """Install required packages"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Successfully installed all requirements")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False
    return True

def check_knowledge_base():
    """Check if PDFs are present in knowledge_base folder"""
    kb_path = Path("knowledge_base")
    pdf_files = list(kb_path.glob("*.pdf"))
    
    if not pdf_files:
        print("⚠️  No PDF files found in knowledge_base folder")
        print("   Please add your legal knowledge base PDFs to the 'knowledge_base' folder")
        print("   Expected: 899, 508, and 140 page PDFs with legal drafting information")
        return False
    
    print(f"✅ Found {len(pdf_files)} PDF files in knowledge_base:")
    for pdf in pdf_files:
        print(f"   - {pdf.name}")
    return True

def create_env_template():
    """Create a template .env file"""
    env_content = """# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Other configurations
STREAMLIT_SERVER_PORT=8501
"""
    
    env_path = Path(".env")
    if not env_path.exists():
        with open(env_path, "w") as f:
            f.write(env_content)
        print("✅ Created .env template file")
        print("   Please add your OpenAI API key to the .env file")
    else:
        print("✅ .env file already exists")

def main():
    print("🚀 Setting up Legal Document Generator...")
    print("=" * 50)
    
    # # Create directory structure
    # print("\n📁 Creating directory structure...")
    # create_directory_structure()
    
    # # Install requirements
    # print("\n📦 Installing Python packages...")
    # if not install_requirements():
    #     print("❌ Setup failed during package installation")
    #     return
    
    # Check knowledge base
    print("\n📚 Checking knowledge base...")
    check_knowledge_base()
    
    # Create environment template
    print("\n🔧 Setting up environment...")
    create_env_template()
    
    print("\n" + "=" * 50)
    print("✅ Setup completed!")
    print("\n📋 Next steps:")
    print("1. Add your PDF files to the 'knowledge_base' folder")
    print("2. Add your OpenAI API key to the .env file or enter it in the app")
    print("3. Run the application: streamlit run legal_doc_generator.py")
    print("\n💡 For help with LaTeX to PDF conversion:")
    print("   - Use Overleaf (online): https://overleaf.com")
    print("   - Install LaTeX locally: https://www.latex-project.org/get/")
    print("   - Use pandoc for DOCX conversion: https://pandoc.org")

if __name__ == "__main__":
    main()