import pdfplumber
import ollama
import requests
from typing import List, Dict

# Load the Llama 3.1 model via Ollama
model = ollama.load("llama3.1:latest")

# Step 1: Extract text from PDF and create a glossary dictionary
def extract_glossary_from_pdf(pdf_path: str) -> Dict[str, str]:
    glossary = {}
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            lines = text.split('\n')
            for line in lines:
                if ':' in line:  # Assuming format "Term: Definition"
                    term, definition = line.split(':', 1)
                    glossary[term.strip()] = definition.strip()
    return glossary

# Step 2: Define a function to retrieve definitions using RAG
def retrieve_definition(term: str, glossary: Dict[str, str]) -> str:
    # Check if term exists in the glossary
    if term in glossary:
        return glossary[term]
    
    # If not found, use the model to generate a definition
    response = model.chat(
        messages=[{
            'role': 'user',
            'content': f"Provide a medical definition for '{term}'."
        }]
    )
    return response['message']['content']

# Main function to run the mapping process
def main():
    # Path to your PDF file
    pdf_path = "159358_AMAGlossaryofMedicalTerms_Ver1.0.pdf"

    # Step 1: Extract glossary from PDF
    print("Extracting glossary from PDF...")
    glossary = extract_glossary_from_pdf(pdf_path)

    # Example terms to map (you can replace these with actual terms extracted)
    terms_to_map = ["Diabetes", "Hypertension", "Cancer"]

    # Step 2: Map definitions for each term
    print("\nMapping definitions...")
    for term in terms_to_map:
        definition = retrieve_definition(term, glossary)
        print(f"{term}: {definition}")

if __name__ == "__main__":
    main()
