from typing import Any
import spacy
import warnings
from itertools import combinations
import networkx as nx
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Suppress common warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Load the spaCy model once (as global)
nlp = spacy.load("en_core_web_trf")

# Initialize rich console
console = Console()


def read_file(path: str) -> str:
    """Read a text file and return its content as a string."""
    text: str
    with open(path, "r", encoding="utf-8") as file:
        text = file.read()
    text = " ".join(text.split())
    return text


def process_text(text: str) -> Any:
    """
    Process text with spaCy and return the Doc object.
    """
    doc = nlp(text)
    return doc


if __name__ == "__main__":
    file_path: str = "data/raw/euthyphro.txt"
    document_text: str = read_file(file_path)

    # Process the text with spaCy
    doc = process_text(document_text)

    # Extract and print sentences
    console.print(Panel("Sentences extracted from the document", style="bold blue"))
    
    for i, sentence in enumerate(doc.sents, 1):
        console.print(f"[bold cyan]Sentence {i}:[/bold cyan] {sentence.text}")
        console.print()
    
    # Create lists to store all entities and co-occurrence pairs
    all_entities = []
    co_occurrence_pairs = []
    
    # Create an empty NetworkX graph
    G = nx.Graph()
    
    console.print(Panel("Building Graph: Processing sentences for person entity pairs", style="bold green"))
    
    # Process each sentence to find co-occurrences
    for i, sentence in enumerate(doc.sents, 1):
        # Process the sentence to get its entities
        sentence_doc = nlp(sentence.text)
        
        # Extract PERSON entities from this sentence
        person_entities = [ent.text for ent in sentence_doc.ents if ent.label_ == "PERSON"]
        
        # Get unique person entities using set()
        unique_persons = set(person_entities)
        
        # Add person entities to the all_entities list
        all_entities.extend(person_entities)
        
        # Add each entity as a node to the graph
        for entity in unique_persons:
            G.add_node(entity)
        
        # Generate all pairs of names in this sentence (co-occurrences)
        if len(unique_persons) >= 2:
            sentence_pairs = list(combinations(unique_persons, 2))
            co_occurrence_pairs.extend(sentence_pairs)
            
            # Add edges to the graph for each pair
            for entity_1, entity_2 in sentence_pairs:
                G.add_edge(entity_1, entity_2)
            
            # Print sentence and its co-occurrence pairs
            console.print(f"[bold cyan]Sentence {i}:[/bold cyan] {sentence.text[:100]}...")
            console.print(f"[yellow]Person entities:[/yellow] {list(unique_persons)}")
            console.print(f"[green]Co-occurrence pairs:[/green] {sentence_pairs}")
            console.print()
    
    # Print summary
    console.print(Panel("Graph Analysis Summary", style="bold magenta"))
    console.print(f"[green]Total person entities found:[/green] {len(all_entities)}")
    console.print(f"[green]Total co-occurrence pairs:[/green] {len(co_occurrence_pairs)}")
    console.print(f"[green]Graph nodes (unique entities):[/green] {G.number_of_nodes()}")
    console.print(f"[green]Graph edges (connections):[/green] {G.number_of_edges()}")
    
    console.print(f"\n[yellow]All nodes in graph:[/yellow]")
    for node in G.nodes():
        console.print(f"  {node}")
    
    console.print(f"\n[yellow]All edges with weights:[/yellow]")
    for edge in G.edges(data=True):
        entity_1, entity_2, data = edge
        weight = data.get('weight', 1)
        console.print(f"  {entity_1} -- {entity_2} (weight: {weight})")