"""
Configuration management for the sentiment analysis system.
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
import json


@dataclass
class SentimentConfig:
    """Configuration for sentiment analysis."""
    
    # Model settings
    default_model: str = "cardiffnlp/twitter-roberta-base-sentiment-latest"
    device: Optional[str] = None  # Auto-detect
    batch_size: int = 32
    max_length: int = 512
    cache_dir: Optional[str] = None
    
    # Analysis settings
    confidence_threshold: float = 0.8
    enable_batch_processing: bool = True
    
    # Logging
    log_level: str = "INFO"
    
    # Model registry for easy switching
    model_registry: Dict[str, str] = field(default_factory=lambda: {
        "roberta": "cardiffnlp/twitter-roberta-base-sentiment-latest",
        "distilbert": "distilbert-base-uncased-finetuned-sst-2-english",
        "finbert": "ProsusAI/finbert",
        "bertweet": "vinai/bertweet-base",
        "vader_roberta": "cardiffnlp/twitter-roberta-base-sentiment"
    })
    
    @classmethod
    def from_file(cls, config_path: str) -> 'SentimentConfig':
        """Load configuration from JSON file."""
        if not os.path.exists(config_path):
            return cls()
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        return cls(**config_data)
    
    def to_file(self, config_path: str) -> None:
        """Save configuration to JSON file."""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(self.__dict__, f, indent=2)
    
    def get_model_name(self, model_key: str) -> str:
        """Get full model name from registry key."""
        return self.model_registry.get(model_key, model_key)


def load_config(config_path: Optional[str] = None) -> SentimentConfig:
    """
    Load configuration from file or environment variables.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        SentimentConfig instance
    """
    if config_path is None:
        # Look for config in standard locations
        possible_paths = [
            "config/sentiment.json",
            ".config/sentiment.json",
            os.path.expanduser("~/.nexus/sentiment.json")
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                config_path = path
                break
    
    if config_path and os.path.exists(config_path):
        config = SentimentConfig.from_file(config_path)
    else:
        config = SentimentConfig()
    
    # Override with environment variables if present
    if os.getenv("SENTIMENT_MODEL"):
        config.default_model = os.getenv("SENTIMENT_MODEL")
    
    if os.getenv("SENTIMENT_DEVICE"):
        config.device = os.getenv("SENTIMENT_DEVICE")
    
    if os.getenv("SENTIMENT_BATCH_SIZE"):
        config.batch_size = int(os.getenv("SENTIMENT_BATCH_SIZE"))
    
    if os.getenv("SENTIMENT_CACHE_DIR"):
        config.cache_dir = os.getenv("SENTIMENT_CACHE_DIR")
    
    return config