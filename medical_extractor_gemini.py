import os
import google.generativeai as genai
from langchain_chroma import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

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

        # Load extracted glossary terms
        self.medical_glossary = self._extract_medical_glossary()
        self.glossary_terms = set(term.lower() for term in self.medical_glossary.split("\n") if term.strip())

        # Initialize vector store
        self.vector_store = self._load_or_create_vector_store()

    def _extract_medical_glossary(self) -> str:
        """Extracts medical terms from a PDF glossary using LangChain's PyPDFLoader."""
        loader = PyPDFLoader(self.pdf_path)
        pages = loader.load()
        
        # Extract text from each page
        glossary_text = [page.page_content for page in pages]
        return "\n".join(glossary_text)
    
    def _load_or_create_vector_store(self):
        """Loads or creates a vector store for medical glossary terms using GoogleAIEmbeddings."""
        
        embedding_function = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=api_key  # ✅ Correct API Key
        )

        # ✅ Define a collection name (important for managing stored data)
        collection_name = "medical_glossary"

        # ✅ If the vector store exists, load it
        if os.path.exists(self.persist_directory):
            return Chroma(
                collection_name=collection_name,
                embedding_function=embedding_function,
                persist_directory=self.persist_directory
            )

        # ✅ Split text into chunks
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(self.medical_glossary)
        docs = [Document(page_content=chunk) for chunk in chunks]

        if not docs:
            raise ValueError("No medical terms found! Check PDF extraction or chunking parameters.")

        # ✅ Correctly initialize Chroma with `collection_name`
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

            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(prompt)

            ontology_mappings[term] = response.text if response else "No mapping found."

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
