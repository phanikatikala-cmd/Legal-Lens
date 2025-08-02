from transformers import pipeline

ner_pipeline = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")

def extract_entities(text):
    entities = ner_pipeline(text)
    return [{"entity": ent['entity_group'], "word": ent['word'], "score": round(ent['score'], 2)} for ent in entities]
