import spacy

nlp = spacy.load("en_core_web_trf")

def process_text(text: str) -> spacy.tokens.Doc:
    """Process text with spaCy and return the Doc object."""
    return nlp(text)