import os
import google.generativeai as genai
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv
import time  # ✅ Add delay between API calls
import re
# ✅ Load environment variables from .env
load_dotenv()

# ✅ Get API key from environment variables
api_key = os.getenv("GEMINI_API_KEY")

# ✅ Ensure API key is correctly loaded
if not api_key:
    raise ValueError("❌ GEMINI_API_KEY is missing! Set it in your .env file.")

# ✅ Configure Gemini API with the correct API key
genai.configure(api_key=api_key)

class MedicalGlossaryExtractor:
    def __init__(self, pdf_path: str, persist_directory: str = "vector_store"):
        self.pdf_path = pdf_path
        self.persist_directory = persist_directory

        # ✅ Extract glossary terms properly
        self.medical_glossary = self._extract_medical_glossary()

        # ✅ Extract only keys (terms) instead of using `.split("\n")`
        self.glossary_terms = set(self.medical_glossary.keys())

        # ✅ Initialize vector store
        self.vector_store = self._load_or_create_vector_store()



    def _extract_medical_glossary(self) -> dict:
        """Extracts medical terms and definitions from the PDF glossary."""
        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load()

        # ✅ Extract text from each page
        glossary_text = [page.page_content for page in pages]
        full_text = "\n".join(glossary_text)

        # ✅ Use regex to extract terms and definitions in "term - definition" format
        pattern = r"([A-Za-z\s\-]+)-\s*([A-Za-z\s\(\)0-9,]+)"
        matches = re.findall(pattern, full_text)

        # ✅ Store terms as a dictionary
        glossary_dict = {term.strip(): definition.strip() for term, definition in matches}
        
        return glossary_dict  # ✅ Returns structured {term: definition} data

    
    def _load_or_create_vector_store(self):
        """Loads or creates a vector store for medical glossary terms using GoogleAIEmbeddings."""
        
        embedding_function = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=api_key
        )

        collection_name = "medical_glossary"

        if os.path.exists(self.persist_directory):
            return Chroma(
                collection_name=collection_name,
                embedding_function=embedding_function,
                persist_directory=self.persist_directory
            )

        # ✅ Use each term + definition as its own chunk
        docs = [Document(page_content=f"{term}: {definition}") for term, definition in self.medical_glossary.items()]

        if not docs:
            raise ValueError("No medical terms found! Check PDF extraction or chunking parameters.")

        # ✅ Correctly initialize Chroma
        vector_store = Chroma.from_documents(
            documents=docs,
            embedding=embedding_function,
            collection_name=collection_name,
            persist_directory=self.persist_directory
        )

        return vector_store


    def extract_medical_terms(self, input_text: str):
        """Extracts medical terms using Gemini 1.5 Flash."""
        if not input_text.strip():
            return "No valid input text provided."

        words = input_text.lower().split()
        matched_terms = [word for word in words if word in self.glossary_terms]

        if matched_terms:
            return matched_terms  # Return terms found directly in the glossary

        # If direct lookup fails, use vector search
        num_indexed_docs = self.vector_store._collection.count()  # ✅ Correct way to count indexed documents
        k_results = min(10, num_indexed_docs)

        if k_results == 0:
            return "Medical glossary database is empty. Check PDF processing."

        similar_docs_with_scores = self.vector_store.similarity_search_with_score(input_text, k=k_results)
        filtered_docs = [doc for doc, score in similar_docs_with_scores if score > 0.7]

        if not filtered_docs:
            return "No medical terms detected."

        context = "\n".join([doc.page_content for doc in filtered_docs])

        # ✅ Use Gemini 1.5 Flash for final extraction
        prompt = f"""
        You are a medical terminology expert. Extract and list only the medical terms **present in the input text**. 

        **Input Text:** {input_text}

        Return only the extracted medical terms in a bullet-point list.
        """

        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)

        return response.text if response else "No medical terms detected."
    
    
    def map_to_ontology(self, extracted_terms):
        """Maps extracted medical terms to ontologies using Gemini 1.5 Flash."""
        if not extracted_terms:
            return "No medical terms to map."

        ontology_mappings = {}

        for term in extracted_terms:
            prompt = f"""
            You are a medical ontology expert. Map the given medical term to relevant medical ontologies 
            such as **ICD-10, SNOMED CT, UMLS, or MeSH**.

            **Medical Term:** {term}

            Provide a structured JSON output with:
            - "ICD-10 Code"
            - "SNOMED CT Code"
            - "UMLS Concept ID"
            - "MeSH Term"
            - "Definition"
            """

            try:
                model = genai.GenerativeModel("gemini-1.5-flash")
                response = model.generate_content(prompt)

                # ✅ Ensure valid JSON output
                ontology_mappings[term] = json.loads(response.text)
                print(f"ontology_mappings[{term}] = ", response.text)

            except Exception as e:
                ontology_mappings[term] = {"error": "API Error"}

        return ontology_mappings

    
if __name__ == "__main__":
    pdf_file = "159358_AMAGlossaryofMedicalTerms_Ver1.0.pdf"
    extractor = MedicalGlossaryExtractor(pdf_file)

    sample_text = "The patient is experiencing severe tachycardia and dyspnea."
    extracted_terms = extractor.extract_medical_terms(sample_text)

    print("\nExtracted Medical Terms:")
    print(extracted_terms)

    ontology_mapping = extractor.map_to_ontology(extracted_terms)

    print("\nOntology Mappings:")
    print(ontology_mapping)
