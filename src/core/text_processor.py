import spacy
from typing import List, Optional


class TextProcessor:
    """Handles text processing using spaCy NLP pipeline."""
    
    def __init__(self, model_name: str = "en_core_web_trf"):
        """
        Initialize the text processor with a spaCy model.
        
        Args:
            model_name: Name of the spaCy model to load
        """
        self.model_name = model_name
        self._nlp = None
    
    @property
    def nlp(self) -> spacy.Language:
        """Lazy load the spaCy model."""
        if self._nlp is None:
            try:
                self._nlp = spacy.load(self.model_name)
            except OSError:
                raise OSError(f"spaCy model '{self.model_name}' not found. "
                            f"Install it with: python -m spacy download {self.model_name}")
        return self._nlp
    
    def process_text(self, text: str) -> spacy.tokens.Doc:
        """
        Process text with spaCy and return the Doc object.
        
        Args:
            text: Input text to process
            
        Returns:
            spaCy Doc object with linguistic annotations
        """
        return self.nlp(text)
    
    def get_sentences(self, text: str) -> List[spacy.tokens.Span]:
        """
        Extract sentences from text.
        
        Args:
            text: Input text to process
            
        Returns:
            List of sentence spans
        """
        doc = self.process_text(text)
        return list(doc.sents)
    
    def get_tokens(self, text: str, include_punct: bool = False) -> List[str]:
        """
        Extract tokens from text.
        
        Args:
            text: Input text to process
            include_punct: Whether to include punctuation tokens
            
        Returns:
            List of token strings
        """
        doc = self.process_text(text)
        if include_punct:
            return [token.text for token in doc]
        else:
            return [token.text for token in doc if not token.is_punct]