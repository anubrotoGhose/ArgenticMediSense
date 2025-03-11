import pdfplumber
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
import ollama

# --- STEP 1: Extract text from PDFs ---
def extract_text_from_pdfs(pdf_paths):
    text_data = []
    for pdf_path in pdf_paths:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:  # Avoid None values
                    text_data.append(extracted_text)
    return "\n".join(text_data)

pdf_files = ["159358_AMAGlossaryofMedicalTerms_Ver1.0.pdf", "Medical-Certificate-Fitness-Certifcate.pdf"]
medical_text = extract_text_from_pdfs(pdf_files)

# --- STEP 2: Prepare embeddings and vector store ---
embedding_model = OllamaEmbeddings(model="mistral:latest")  # Use deepseek-r1 for better embeddings

# ✅ Increased chunk size for better context
text_splitter = CharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = text_splitter.split_text(medical_text)

print(f"Total chunks created: {len(chunks)}")  # Debugging: Ensure multiple chunks are generated
print("\n")
docs = [Document(page_content=chunk) for chunk in chunks]

if len(docs) == 0:
    raise ValueError("No text chunks found! Check PDF extraction or chunking parameters.")

vector_store = Chroma.from_documents(docs, embedding_model)

# --- STEP 3: RAG-based mapping to SNOMED/ICD-10 ---
def query_ontology(term):
    if not term.strip():
        return "No valid medical term provided."

    num_indexed_docs = len(vector_store)
    
    # ✅ Increased retrieval size for better consistency
    k_results = min(5, num_indexed_docs)  # Ensure we don't request more than available

    if k_results == 0:
        return "No medical terms are indexed. Check PDF processing."

    similar_docs = vector_store.similarity_search(term, k=k_results)
    
    if not similar_docs:
        return f"No matching SNOMED-CT or ICD-10 code found for {term}."

    context = "\n".join([doc.page_content for doc in similar_docs])
    
    # ✅ Debugging Step: Print retrieved context to ensure correctness
    # print("\nRetrieved Medical Context:\n", context)

    # ✅ Standardized AI response format
    prompt = f"""
    You are a medical ontology expert. Your task is to return the **best** SNOMED-CT or ICD-10 code for a given medical term.
    Follow this response format strictly:

    **Term:** {term}

    **Best Matching Code:**
    - **SNOMED-CT Code:** [Code] - [Description]
    - **ICD-10 Code:** [Code] - [Description]

    **Explanation:** Provide a brief reason why this code matches.

    If no match is found, return: "No exact match found. Closest related condition: [Condition]."
    
    **Medical Database Context:**
    {context}
    """

    response = ollama.chat(model="thewindmom/llama3-med42-8b", messages=[{"role": "user", "content": prompt}])
    return response["message"]["content"]

# --- Example Usage ---
term = """
**Patient Medical Report**

**Patient Information:**  
- **Name:** John Doe  
- **Age:** 45  
- **Gender:** Male  
- **Date of Birth:** March 10, 1979  
- **Blood Type:** O+  
- **Contact:** (123) 456-7890  
- **Address:** 123 Main Street, City, State, ZIP  

**Medical History:**  
- **Chronic Conditions:** Hypertension, Type 2 Diabetes  
- **Allergies:** Penicillin  
- **Past Surgeries:** Appendectomy (2005)  
- **Family Medical History:** Father - Heart Disease; Mother - Diabetes  

**Current Medications:**  
- Metformin 500mg (Twice Daily)  
- Amlodipine 5mg (Once Daily)  

**Chief Complaint:**  
- Patient presents with persistent chest pain and shortness of breath over the past two weeks.  

**Physical Examination:**  
- **Height:** 5'10"  
- **Weight:** 180 lbs  
- **Blood Pressure:** 140/90 mmHg  
- **Heart Rate:** 82 bpm  
- **Respiratory Rate:** 18 breaths per minute  
- **Temperature:** 98.6°F  

**Diagnostic Tests Ordered:**  
- **Electrocardiogram (ECG)**: To assess heart function.  
- **Echocardiogram**: To evaluate heart structure and motion.  
- **Blood Tests**: Lipid profile, HbA1c, and Troponin levels.  
- **Chest X-ray**: To check for lung or cardiac abnormalities.  

**Assessment & Plan:**  
- Suspected mild angina; awaiting test results for confirmation.  
- Continue current medications; consider aspirin if cardiac involvement is confirmed.  
- Lifestyle modification: Low-sodium diet, regular exercise, and smoking cessation (if applicable).  
- Follow-up appointment in one week to review test results.  

**Physician's Notes:**  
- Patient advised to seek immediate medical attention if symptoms worsen.  
- Discussed risk factors and necessary lifestyle changes.  

**Physician:** Dr. Jane Smith, MD  
**Date:** February 26, 2025  
**Hospital/Clinic:** City General Hospital  
**Contact:** (987) 654-3210
"""
ontology_mapping = query_ontology(term)
print("\n")
print("The response by AI: ")
print("\n")
print(ontology_mapping)