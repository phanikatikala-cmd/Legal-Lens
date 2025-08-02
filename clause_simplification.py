# clause_simplification.py
from transformers import pipeline

simplify_pipeline = pipeline("summarization", model="facebook/bart-large-cnn")

def simplify_clause(text):
    return simplify_pipeline(text, max_length=100, min_length=30, do_sample=False)[0]['summary_text']
