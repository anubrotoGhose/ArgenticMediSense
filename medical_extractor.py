import pdfplumber
import ollama
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
import os

class MedicalGlossaryExtractor:
    def __init__(self, pdf_path: str, persist_directory: str = "vector_store"):
        self.pdf_path = pdf_path
        self.persist_directory = persist_directory
        self.embedding_model = OllamaEmbeddings(model="llama3.2:latest")
        
        # Extract glossary from PDF
        self.medical_glossary = self._extract_medical_glossary()
        self.glossary_terms = set(term.lower() for term in self.medical_glossary.split("\n") if term.strip())

        # Initialize vector store (load existing or create new)
        self.vector_store = self._load_or_create_vector_store()

    def _extract_medical_glossary(self) -> str:
        """Extracts medical terms from a PDF glossary."""
        glossary_text = []
        with pdfplumber.open(self.pdf_path) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    glossary_text.append(extracted_text)
        return "\n".join(glossary_text)

    def _load_or_create_vector_store(self):
        """Loads existing vector store if available, otherwise creates a new one."""
        if os.path.exists(self.persist_directory):
            return Chroma(persist_directory=self.persist_directory, embedding_function=self.embedding_model)
        
        text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(self.medical_glossary)
        docs = [Document(page_content=chunk) for chunk in chunks]

        if not docs:
            raise ValueError("No medical terms found! Check PDF extraction or chunking parameters.")

        vector_store = Chroma.from_documents(docs, self.embedding_model, persist_directory=self.persist_directory)
        vector_store.persist()  # Save for future use
        return vector_store

    # def extract_medical_terms(self, input_text: str):
    #     """Extracts medical terms from input text using dictionary lookup & vector search."""
    #     if not input_text.strip():
    #         return "No valid input text provided."

    #     words = input_text.lower().split()
    #     matched_terms = [word for word in words if word in self.glossary_terms]

    #     if matched_terms:
    #         return matched_terms  # Return terms found directly in the glossary

    #     # If direct lookup fails, use vector search
    #     num_indexed_docs = len(self.vector_store)
    #     k_results = min(10, num_indexed_docs)

    #     if k_results == 0:
    #         return "Medical glossary database is empty. Check PDF processing."

    #     similar_docs_with_scores = self.vector_store.similarity_search_with_score(input_text, k=k_results)
    #     filtered_docs = [doc for doc, score in similar_docs_with_scores if score > 0.7]

    #     if not filtered_docs:
    #         return "No medical terms detected."

    #     context = "\n".join([doc.page_content for doc in filtered_docs])

    #     # Use Ollama for final extraction
    #     prompt = f"""
    #     You are a medical terminology expert. Extract and list only the medical terms **present in the input text**. 

    #     **Input Text:** {input_text}

    #     Return only the extracted medical terms in a bullet-point list.
    #     """

    #     response = ollama.chat(model="thewindmom/llama3-med42-8b", messages=[{"role": "user", "content": prompt}])
    #     return response["message"]["content"]

    # def map_to_ontology(self, extracted_terms):
    #     """Maps extracted medical terms to ontologies using RAG-based retrieval."""
    #     if not extracted_terms:
    #         return "No medical terms to map."

    #     ontology_mappings = {}

    #     for term in extracted_terms:
    #         # search_results = self.vector_store.similarity_search_with_score(term, k=5)
    #         # relevant_contexts = [doc.page_content for doc, score in search_results if score > 0.6]

    #         # if not relevant_contexts:
    #         #     ontology_mappings[term] = "No ontology mapping found."
    #         #     continue

    #         # Generate ontology mapping using RAG
    #         prompt = f"""
    #         You are a medical ontology expert. Map the given medical term to relevant medical ontologies 
    #         such as **ICD-10, SNOMED CT, UMLS, or MeSH**.

    #         **Medical Term:** {term}


    #         Provide a structured JSON output with:
    #         - "ICD-10 Code"
    #         - "SNOMED CT Code"
    #         - "UMLS Concept ID"
    #         - "MeSH Term"
    #         - "Definition"
    #         """
    #         # Generate ontology mapping using RAG
    #         # prompt = f"""
    #         # You are a medical ontology expert. Map the given medical term to relevant medical ontologies 
    #         # such as **ICD-10, SNOMED CT, UMLS, or MeSH**.

    #         # **Medical Term:** {term}

    #         # **Reference Context:**
    #         # {''.join(relevant_contexts)}

    #         # Provide a structured JSON output with:
    #         # - "ICD-10 Code"
    #         # - "SNOMED CT Code"
    #         # - "UMLS Concept ID"
    #         # - "MeSH Term"
    #         # - "Definition"
    #         # """
    #         response = ollama.chat(model="thewindmom/llama3-med42-8b", messages=[{"role": "user", "content": prompt}])
    #         ontology_mappings[term] = response["message"]["content"]

    #     return ontology_mappings

    def extract_medical_terms(self, input_text: str):
        """Extracts medical terms from input text using dictionary lookup & vector search."""
        if not input_text.strip():
            return "No valid input text provided."

        words = input_text.lower().split()
        matched_terms = [word for word in words if word in self.glossary_terms]

        if matched_terms:
            return matched_terms  # Return terms found directly in the glossary

        # If direct lookup fails, use vector search
        num_indexed_docs = len(self.vector_store)
        k_results = min(10, num_indexed_docs)

        if k_results == 0:
            return "Medical glossary database is empty. Check PDF processing."

        similar_docs_with_scores = self.vector_store.similarity_search_with_score(input_text, k=k_results)
        filtered_docs = [doc for doc, score in similar_docs_with_scores if score > 0.7]

        if not filtered_docs:
            return "No medical terms detected."

        context = "\n".join([doc.page_content for doc in filtered_docs])

        # Use Ollama for final extraction
        prompt = f"""
        You are a medical terminology expert. Extract and list only the medical terms **present in the input text**. 

        **Input Text:** {input_text}

        Return only the extracted medical terms in a bullet-point list.
        """

        response = ollama.chat(model="thewindmom/llama3-med42-8b", messages=[{"role": "user", "content": prompt}])
        return response["message"]["content"]

    def map_to_ontology(self, extracted_terms):
        """Maps extracted medical terms to ontologies using RAG-based retrieval."""
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
            model_name = "llama3.2:latest"
            response = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
            ontology_mappings[term] = response["message"]["content"]

        return ontology_mappings


# --- Example Usage (Only runs when executed directly) ---
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