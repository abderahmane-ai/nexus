from itertools import combinations
import networkx as nx
from typing import List, Tuple, Set, Dict
import spacy
from collections import Counter, defaultdict
import logging
from ..core.entity_extractor import EntityExtractor
from ..core.sentiment_analyzer import SentimentAnalyzer


class GraphBuilder:
    """Builds network graphs from text by analyzing entity co-occurrences."""
    
    def __init__(self, entity_extractor: EntityExtractor = None, use_sentiment: bool = True):
        """
        Initialize the graph builder.
        
        Args:
            entity_extractor: EntityExtractor instance to use for entity extraction
            use_sentiment: Whether to analyze sentiment for entity relationships
        """
        self.entity_extractor = entity_extractor or EntityExtractor()
        self.use_sentiment = use_sentiment
        self.sentiment_analyzer = SentimentAnalyzer() if use_sentiment else None
        self.logger = logging.getLogger(__name__)
    
    def get_major_entities(
        self, 
        sentences: List[spacy.tokens.Span], 
        min_mentions: int = 3, 
        verbose: bool = True
    ) -> Set[str]:
        """
        Identify major entities based on frequency of mentions.
        
        Args:
            sentences: List of spaCy sentence spans
            min_mentions: Minimum number of mentions required to be considered major
            verbose: Whether to print analysis output
            
        Returns:
            Set of entity names that appear at least min_mentions times
        """
        entity_counts: Counter[str] = Counter()
        
        for sentence in sentences:
            unique_entities = self.entity_extractor.get_unique_entities(sentence)
            for entity in unique_entities:
                entity_counts[entity] += 1
        
        # Filter entities that appear at least min_mentions times
        major_entities = {entity for entity, count in entity_counts.items() if count >= min_mentions}
        
        # Log for debugging
        self.logger.debug(f"Found {len(entity_counts)} total entities, {len(major_entities)} major entities")
        
        if verbose:
            print(f"\n🎭 Entity Analysis:")
            print(f"   Total entities: {len(entity_counts)} | Major entities (≥{min_mentions} mentions): {len(major_entities)}")
            
            # Show only top 10 entities
            sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
            print(f"   Top entities:")
            for entity, count in sorted_entities[:10]:
                status = "📊" if entity in major_entities else "📝"
                print(f"     {status} {entity}: {count}")
        
        return major_entities
    
    def build_graph_from_sentences(
        self, 
        sentences: List[spacy.tokens.Span], 
        min_mentions: int = 3, 
        verbose: bool = True
    ) -> nx.Graph:
        """
        Build a network graph from sentences by extracting co-occurring entities.
        
        Only includes major entities (those mentioned at least min_mentions times) in the visualization,
        but calculates full connection counts including connections with minor entities.
        When sentiment analysis is enabled, calculates average sentiment for each relationship.
        
        Args:
            sentences: List of spaCy sentence spans to analyze
            min_mentions: Minimum mentions required for an entity to be included in graph
            verbose: Whether to print analysis output
            
        Returns:
            NetworkX graph with major entities as nodes and co-occurrences as edges
        """
        # First, identify major entities
        major_entities = self.get_major_entities(sentences, min_mentions, verbose)
        
        # Calculate full connection counts for all entities (including connections with minor entities)
        full_connections: Counter[str] = Counter()
        major_to_major_connections: Counter[Tuple[str, str]] = Counter()
        
        # Track sentiment data for major-to-major connections
        sentiment_data: Dict[Tuple[str, str], List[float]] = defaultdict(list)
        
        if verbose and self.use_sentiment:
            print(f"\n💭 Analyzing sentiment for entity relationships...")
        
        for sentence in sentences:
            unique_entities = self.entity_extractor.get_unique_entities(sentence)
            
            # Count all connections for each entity (including with minor entities)
            if len(unique_entities) >= 2:
                pairs = list(combinations(unique_entities, 2))
                
                # Analyze sentiment for this sentence if it contains entity pairs
                sentence_sentiment = None
                if self.use_sentiment and len(unique_entities) >= 2:
                    try:
                        result = self.sentiment_analyzer.analyze(sentence.text)
                        # Convert sentiment to numerical score (-1 to 1)
                        sentence_sentiment = self._convert_sentiment_to_score(result)
                    except Exception as e:
                        self.logger.warning(f"Sentiment analysis failed for sentence: {e}")
                        sentence_sentiment = 0.0  # Neutral fallback
                
                for entity_1, entity_2 in pairs:
                    # Count full connections for both entities
                    full_connections[entity_1] += 1
                    full_connections[entity_2] += 1
                    
                    # Also track major-to-major connections separately
                    if entity_1 in major_entities and entity_2 in major_entities:
                        pair = tuple(sorted([entity_1, entity_2]))
                        major_to_major_connections[pair] += 1
                        
                        # Store sentiment data for this relationship
                        if sentence_sentiment is not None:
                            sentiment_data[pair].append(sentence_sentiment)
        
        # Create the graph
        G = nx.Graph()
        
        # Add major entities to the graph with their full connection counts
        for entity in major_entities:
            G.add_node(entity, full_connections=full_connections[entity])
        
        # Add edges between major entities with their co-occurrence weights and sentiment
        for (entity_1, entity_2), weight in major_to_major_connections.items():
            edge_attrs = {'weight': weight}
            
            # Calculate average sentiment for this relationship
            if self.use_sentiment and (entity_1, entity_2) in sentiment_data:
                sentiments = sentiment_data[(entity_1, entity_2)]
                if sentiments:
                    avg_sentiment = sum(sentiments) / len(sentiments)
                    edge_attrs['sentiment'] = round(avg_sentiment, 3)
                    edge_attrs['sentiment_count'] = len(sentiments)
                else:
                    edge_attrs['sentiment'] = 0.0
                    edge_attrs['sentiment_count'] = 0
            
            G.add_edge(entity_1, entity_2, **edge_attrs)
        
        if verbose and self.use_sentiment:
            sentiment_edges = [data for _, _, data in G.edges(data=True) if 'sentiment' in data]
            if sentiment_edges:
                avg_sentiment = sum(data['sentiment'] for data in sentiment_edges) / len(sentiment_edges)
                print(f"   📊 Average relationship sentiment: {avg_sentiment:.3f}")
        
        self.logger.debug(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G
    
    def get_entity_statistics(self, sentences: List[spacy.tokens.Span]) -> Dict[str, int]:
        """
        Get statistics about entity occurrences.
        
        Args:
            sentences: List of spaCy sentence spans
            
        Returns:
            Dictionary mapping entity names to occurrence counts
        """
        entity_counts: Counter[str] = Counter()
        
        for sentence in sentences:
            unique_entities = self.entity_extractor.get_unique_entities(sentence)
            for entity in unique_entities:
                entity_counts[entity] += 1
        
        return dict(entity_counts)
    
    def _convert_sentiment_to_score(self, sentiment_result) -> float:
        """
        Convert sentiment analysis result to numerical score (-1 to 1).
        
        Args:
            sentiment_result: SentimentResult from sentiment analyzer
            
        Returns:
            Float score where -1 is very negative, 0 is neutral, 1 is very positive
        """
        label = sentiment_result.label.lower()
        confidence = sentiment_result.confidence
        
        # Map common sentiment labels to scores
        positive_labels = ['positive', 'joy', 'love', 'optimism', 'admiration', 'approval', 'caring']
        negative_labels = ['negative', 'anger', 'sadness', 'fear', 'disgust', 'disapproval', 'annoyance']
        neutral_labels = ['neutral', 'surprise', 'curiosity', 'confusion', 'realization']
        
        if any(pos in label for pos in positive_labels):
            return confidence  # Positive score
        elif any(neg in label for neg in negative_labels):
            return -confidence  # Negative score
        else:
            return 0.0  # Neutral score