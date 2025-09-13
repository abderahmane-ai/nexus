from itertools import combinations
import networkx as nx
from typing import List, Tuple, Set
import spacy
from collections import Counter
from .entity_extractor import get_unique_entities

def get_major_entities(sentences: List[spacy.tokens.Span], min_mentions: int = 3) -> Set[str]:
    """Identify major entities based on frequency of mentions."""
    entity_counts = Counter()
    
    for sentence in sentences:
        unique_entities = get_unique_entities(sentence)
        for entity in unique_entities:
            entity_counts[entity] += 1
    
    # Filter entities that appear at least min_mentions times
    major_entities = {entity for entity, count in entity_counts.items() if count >= min_mentions}
    
    print(f"\n🎭 Entity Analysis:")
    print(f"   Total unique entities found: {len(entity_counts)}")
    print(f"   Major entities (≥{min_mentions} mentions): {len(major_entities)}")
    
    # Show all entities with their counts
    sorted_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)
    print(f"   All entities by frequency:")
    for entity, count in sorted_entities:
        status = "📊 MAJOR" if entity in major_entities else "📝 minor"
        print(f"     • {entity}: {count} mentions {status}")
    
    return major_entities

def build_graph_from_sentences(sentences: List[spacy.tokens.Span], min_mentions: int = 3) -> nx.Graph:
    """Build a network graph from sentences by extracting co-occurring PERSON entities.
    Only includes major entities (those mentioned at least min_mentions times) in the visualization,
    but calculates full connection counts including connections with minor entities.
    """
    # First, identify major entities
    major_entities = get_major_entities(sentences, min_mentions)
    
    # Calculate full connection counts for all entities (including connections with minor entities)
    full_connections = Counter()
    major_to_major_connections = Counter()
    
    for sentence in sentences:
        unique_entities = get_unique_entities(sentence)
        
        # Count all connections for each entity (including with minor entities)
        if len(unique_entities) >= 2:
            pairs = list(combinations(unique_entities, 2))
            for entity_1, entity_2 in pairs:
                # Count full connections for both entities
                full_connections[entity_1] += 1
                full_connections[entity_2] += 1
                
                # Also track major-to-major connections separately
                if entity_1 in major_entities and entity_2 in major_entities:
                    pair = tuple(sorted([entity_1, entity_2]))
                    major_to_major_connections[pair] += 1
    
    G = nx.Graph()
    
    # Add major entities to the graph with their full connection counts
    for entity in major_entities:
        G.add_node(entity, full_connections=full_connections[entity])
    
    # Add edges between major entities with their co-occurrence weights
    for (entity_1, entity_2), weight in major_to_major_connections.items():
        G.add_edge(entity_1, entity_2, weight=weight)
    
    return G