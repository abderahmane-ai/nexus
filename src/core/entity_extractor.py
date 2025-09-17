from typing import List, Set
import spacy


class EntityExtractor:
    """Handles extraction of named entities from text using spaCy."""
    
    def __init__(self, entity_types: List[str] = None):
        """
        Initialize the entity extractor.
        
        Args:
            entity_types: List of entity types to extract (default: ["PERSON"])
        """
        self.entity_types = entity_types or ["PERSON"]
    
    def extract_entities_from_sentence(self, sentence: spacy.tokens.Span) -> List[str]:
        """Extract entities of specified types from a spaCy sentence span."""
        return [ent.text for ent in sentence.ents if ent.label_ in self.entity_types]
    
    def get_unique_entities(self, sentence: spacy.tokens.Span) -> Set[str]:
        """
        Get unique entities from a sentence.
        
        Args:
            sentence: spaCy sentence span
            
        Returns:
            Set of unique entity names
        """
        entities = self.extract_entities_from_sentence(sentence)
        return set(entities)
    
    def extract_all_entities(self, sentences: List[spacy.tokens.Span]) -> List[Set[str]]:
        """
        Extract unique entities from multiple sentences.
        
        Args:
            sentences: List of spaCy sentence spans
            
        Returns:
            List of sets, each containing unique entities from a sentence
        """
        return [self.get_unique_entities(sentence) for sentence in sentences]