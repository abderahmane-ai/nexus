from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import networkx as nx

console = Console()

def print_sentences(sentences: list, limit: int = 3) -> None:
    """Print the first few sentences with rich formatting."""
    console.print(Panel("📄 Document Preview", style="bold blue"))
    for i, sentence in enumerate(sentences[:limit], 1):
        console.print(f"[cyan]{i}.[/cyan] {sentence.text[:100]}{'...' if len(sentence.text) > 100 else ''}")

def print_graph_summary(G: nx.Graph, sentences: list) -> None:
    """Print a concise summary of the graph using rich."""
    from src.entity_extractor import get_unique_entities
    
    # Calculate total entities from sentences
    all_entities = []
    for sentence in sentences:
        unique_entities = get_unique_entities(sentence)
        all_entities.extend(unique_entities)
    
    console.print(Panel("📊 Network Summary", style="bold magenta"))
    console.print(f"[green]Nodes:[/green] {G.number_of_nodes()} | [green]Connections:[/green] {G.number_of_edges()} | [green]Density:[/green] {nx.density(G):.3f}")

    if G.number_of_nodes() > 0:
        # Show top 5 most connected entities
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
        console.print(f"\n[yellow]Most connected:[/yellow] " + " | ".join([f"{node}: {degree}" for node, degree in top_nodes]))
    else:
        console.print(f"\n[red]No major entities found. Try lowering --min-mentions.[/red]")