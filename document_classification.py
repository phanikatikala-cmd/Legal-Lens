
# document_classification.py
from transformers import pipeline

classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

LABELS = ["Non-Disclosure Agreement", "Lease Agreement", "Employment Contract", "Service Agreement", "Partnership Agreement"]

def classify_document(text):
    result = classifier(text, candidate_labels=LABELS)
    return result['labels'][0]