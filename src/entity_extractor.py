from typing import List, Set
import spacy

def extract_entities_from_sentence(sentence: spacy.tokens.Span) -> List[str]:
    """Extract all PERSON entities from a spaCy sentence span."""
    return [ent.text for ent in sentence.ents if ent.label_ == "PERSON"]

def get_unique_entities(sentence: spacy.tokens.Span) -> Set[str]:
    """Get unique PERSON entities from a sentence."""
    entities = extract_entities_from_sentence(sentence)
    return set(entities)