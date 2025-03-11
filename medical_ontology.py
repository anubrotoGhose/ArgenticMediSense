from medical_extractor import MedicalGlossaryExtractor

# Initialize once
pdf_file = "159358_AMAGlossaryofMedicalTerms_Ver1.0.pdf"
extractor = MedicalGlossaryExtractor(pdf_file)

# Use in different scripts
input_text = "The patient is suffering from severe tachycardia and dyspnea."
extracted_terms = extractor.extract_medical_terms(input_text)

print("\nExtracted Medical Terms:")
print(extracted_terms)
