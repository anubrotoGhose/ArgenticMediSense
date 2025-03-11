import pdfplumber
import ollama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document

# --- STEP 1: Extract Medical Terms from PDF Glossary ---
def extract_medical_glossary(pdf_path):
    glossary_text = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            extracted_text = page.extract_text()
            if extracted_text:  
                glossary_text.append(extracted_text)
    return "\n".join(glossary_text)

pdf_file = "159358_AMAGlossaryofMedicalTerms_Ver1.0.pdf"
medical_glossary = extract_medical_glossary(pdf_file)

# --- STEP 2: Prepare Embeddings and Vector Store ---
embedding_model = OllamaEmbeddings(model="llama3.2:latest")

text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_text(medical_glossary)

print(f"Total glossary chunks created: {len(chunks)}")  # Debugging check

docs = [Document(page_content=chunk) for chunk in chunks]

if len(docs) == 0:
    raise ValueError("No medical terms found! Check PDF extraction or chunking parameters.")

vector_store = Chroma.from_documents(docs, embedding_model)

# --- STEP 3: Dictionary Lookup for Fast Extraction ---
glossary_terms = set(term.lower() for term in medical_glossary.split("\n") if term.strip())

def extract_medical_terms(input_text):
    if not input_text.strip():
        return "No valid input text provided."

    words = input_text.lower().split()
    matched_terms = [word for word in words if word in glossary_terms]

    if matched_terms:
        return matched_terms  # Return terms found directly in the glossary

    # --- STEP 4: Use Similarity Search if Dictionary Fails ---
    num_indexed_docs = len(vector_store)
    k_results = min(10, num_indexed_docs)  

    if k_results == 0:
        return "Medical glossary database is empty. Check PDF processing."

    similar_docs_with_scores = vector_store.similarity_search_with_score(input_text, k=k_results)
    filtered_docs = [doc for doc, score in similar_docs_with_scores if score > 0.7]  # Score threshold

    if not filtered_docs:
        return "No medical terms detected."

    context = "\n".join([doc.page_content for doc in filtered_docs])

    # --- STEP 5: Query Ollama for Medical Terms Extraction ---
    prompt = f"""
    You are a medical terminology expert. Extract and list only the medical terms **present in the input text**. 

    **Input Text:** {input_text}



    Return only the extracted medical terms in a bullet-point list.
    """

    response = ollama.chat(model="thewindmom/llama3-med42-8b", messages=[{"role": "user", "content": prompt}])
    
    return response["message"]["content"]

# --- Example Usage ---
input_text = "The patient is experiencing severe hypertension and wheezing. Possible signs of asthma."
extracted_terms = extract_medical_terms(input_text)

print("\nExtracted Medical Terms:")
print(extracted_terms)