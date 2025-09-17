"""
Professional sentiment analysis module with configurable models and comprehensive analysis.
"""

import logging
from typing import Dict, List, Optional, Union, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import warnings

import numpy as np
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification


class SentimentModel(Enum):
    """Available sentiment analysis models."""
    
    # High-performance models for historical texts
    EMOTION_ROBERTA = "j-hartmann/emotion-english-distilroberta-base"  # 7 emotions
    GO_EMOTIONS = "SamLowe/roberta-base-go_emotions"  # 28 emotions
    MULTILINGUAL_SENTIMENT = "nlptown/bert-base-multilingual-uncased-sentiment"
    
    # State-of-the-art performance
    DEBERTA_V3 = "microsoft/deberta-v3-base"
    
    # For longer historical documents
    LONGFORMER_SENTIMENT = "allenai/longformer-base-4096"


@dataclass
class SentimentResult:
    """Structured sentiment analysis result."""
    text: str
    label: str
    confidence: float
    scores: Dict[str, float]
    model_used: str
    
    def __post_init__(self):
        """Validate and normalize the result."""
        self.confidence = round(self.confidence, 4)
        self.scores = {k: round(v, 4) for k, v in self.scores.items()}


class SentimentAnalyzer:
    """
    Professional sentiment analyzer with multiple model support and comprehensive analysis.
    """
    
    def __init__(
        self,
        model_name: Union[str, SentimentModel] = SentimentModel.GO_EMOTIONS,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_name: Model to use for sentiment analysis
            device: Device to run inference on ('cpu', 'cuda', 'mps', or None for auto)
            batch_size: Batch size for processing multiple texts
            max_length: Maximum sequence length for tokenization
            cache_dir: Directory to cache downloaded models
        """
        self.model_name = model_name.value if isinstance(model_name, SentimentModel) else model_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache_dir = cache_dir
        
        self.logger = logging.getLogger(__name__)
        self._pipeline = None
        self._tokenizer = None
        
        # Suppress warnings for cleaner output
        warnings.filterwarnings("ignore", category=UserWarning)
        
    def _initialize_pipeline(self) -> None:
        """Lazy initialization of the sentiment analysis pipeline."""
        if self._pipeline is None:
            try:
                self.logger.info(f"Loading sentiment model: {self.model_name}")
                self._pipeline = pipeline(
                    "sentiment-analysis",
                    model=self.model_name,
                    device=self.device,
                    model_kwargs={"cache_dir": self.cache_dir} if self.cache_dir else {},
                    truncation=True,
                    max_length=self.max_length
                )
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir
                )
                self.logger.info("Sentiment model loaded successfully")
            except Exception as e:
                self.logger.error(f"Failed to load model {self.model_name}: {e}")
                raise
    
    def analyze(self, text: Union[str, List[str]]) -> Union[SentimentResult, List[SentimentResult]]:
        """
        Analyze sentiment of text(s).
        
        Args:
            text: Single text string or list of texts to analyze
            
        Returns:
            SentimentResult or list of SentimentResults
        """
        self._initialize_pipeline()
        
        is_single = isinstance(text, str)
        texts = [text] if is_single else text
        
        if not texts or any(not t.strip() for t in texts):
            raise ValueError("Text input cannot be empty or contain only whitespace")
        
        # Process in batches for efficiency
        results = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            batch_results = self._pipeline(batch)
            
            for txt, result in zip(batch, batch_results):
                # Handle different model output formats
                if isinstance(result, list):
                    result = result[0]  # Take the top prediction
                
                # Create comprehensive scores dictionary
                scores = {result['label']: result['score']}
                
                # For models that return all class probabilities
                if hasattr(self._pipeline.model, 'config') and hasattr(self._pipeline.model.config, 'id2label'):
                    try:
                        # Get all class probabilities
                        inputs = self._tokenizer(txt, return_tensors="pt", truncation=True, max_length=self.max_length)
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            outputs = self._pipeline.model(**inputs)
                        
                        probabilities = outputs.logits.softmax(dim=-1).squeeze().tolist()
                        id2label = self._pipeline.model.config.id2label
                        scores = {id2label[i]: prob for i, prob in enumerate(probabilities)}
                    except Exception:
                        # Fallback to single prediction if full probability extraction fails
                        pass
                
                sentiment_result = SentimentResult(
                    text=txt,
                    label=result['label'],
                    confidence=result['score'],
                    scores=scores,
                    model_used=self.model_name
                )
                results.append(sentiment_result)
        
        return results[0] if is_single else results
    
    def analyze_batch(self, texts: List[str]) -> List[SentimentResult]:
        """
        Analyze sentiment for a batch of texts efficiently.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of SentimentResults
        """
        return self.analyze(texts)
    
    def get_sentiment_distribution(self, texts: List[str]) -> Dict[str, float]:
        """
        Get the distribution of sentiments across multiple texts.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary with sentiment labels as keys and proportions as values
        """
        results = self.analyze_batch(texts)
        
        label_counts = {}
        for result in results:
            label_counts[result.label] = label_counts.get(result.label, 0) + 1
        
        total = len(results)
        return {label: count / total for label, count in label_counts.items()}
    
    def get_confidence_stats(self, texts: List[str]) -> Dict[str, float]:
        """
        Get confidence statistics for sentiment predictions.
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dictionary with confidence statistics
        """
        results = self.analyze_batch(texts)
        confidences = [result.confidence for result in results]
        
        return {
            "mean_confidence": np.mean(confidences),
            "median_confidence": np.median(confidences),
            "std_confidence": np.std(confidences),
            "min_confidence": np.min(confidences),
            "max_confidence": np.max(confidences)
        }
    
    def filter_by_confidence(
        self, 
        texts: List[str], 
        min_confidence: float = 0.8
    ) -> Tuple[List[SentimentResult], List[SentimentResult]]:
        """
        Filter results by confidence threshold.
        
        Args:
            texts: List of texts to analyze
            min_confidence: Minimum confidence threshold
            
        Returns:
            Tuple of (high_confidence_results, low_confidence_results)
        """
        results = self.analyze_batch(texts)
        
        high_confidence = [r for r in results if r.confidence >= min_confidence]
        low_confidence = [r for r in results if r.confidence < min_confidence]
        
        return high_confidence, low_confidence
    
    def compare_models(
        self, 
        text: str, 
        models: List[Union[str, SentimentModel]]
    ) -> Dict[str, SentimentResult]:
        """
        Compare sentiment analysis across different models.
        
        Args:
            text: Text to analyze
            models: List of models to compare
            
        Returns:
            Dictionary mapping model names to their results
        """
        original_model = self.model_name
        results = {}
        
        for model in models:
            try:
                model_name = model.value if isinstance(model, SentimentModel) else model
                self.model_name = model_name
                self._pipeline = None  # Reset pipeline to load new model
                
                result = self.analyze(text)
                results[model_name] = result
            except Exception as e:
                self.logger.warning(f"Failed to analyze with model {model}: {e}")
                results[str(model)] = None
        
        # Restore original model
        self.model_name = original_model
        self._pipeline = None
        
        return results
    
    def __repr__(self) -> str:
        return f"SentimentAnalyzer(model='{self.model_name}', device='{self.device}')"


def create_analyzer(
    model: Union[str, SentimentModel] = SentimentModel.GO_EMOTIONS,
    **kwargs
) -> SentimentAnalyzer:
    """
    Factory function to create a sentiment analyzer with sensible defaults.
    
    Args:
        model: Model to use for sentiment analysis
        **kwargs: Additional arguments passed to SentimentAnalyzer
        
    Returns:
        Configured SentimentAnalyzer instance
    """
    return SentimentAnalyzer(model_name=model, **kwargs)