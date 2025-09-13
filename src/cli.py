from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import networkx as nx

console = Console()

def print_sentences(sentences: list, limit: int = 5) -> None:
    """Print the first few sentences with rich formatting."""
    console.print(Panel("Sentences extracted from the document", style="bold blue"))
    for i, sentence in enumerate(sentences[:limit], 1):
        console.print(f"[bold cyan]Sentence {i}:[/bold cyan] {sentence.text}")
        console.print()

def print_graph_summary(G: nx.Graph, all_entities: list, co_occurrence_pairs: list) -> None:
    """Print a summary of the graph using rich."""
    console.print(Panel("Graph Analysis Summary", style="bold magenta"))
    console.print(f"[green]Total person entities found in text:[/green] {len(set(all_entities))}")
    console.print(f"[green]Major entities in visualization:[/green] {G.number_of_nodes()}")
    console.print(f"[green]Connections between major entities:[/green] {G.number_of_edges()}")
    console.print(f"[green]Total co-occurrence pairs (all entities):[/green] {len(co_occurrence_pairs)}")

    if G.number_of_nodes() > 0:
        console.print(f"\n[yellow]Major entities with full connection counts:[/yellow]")
        for node in G.nodes():
            full_conn = G.nodes[node].get('full_connections', G.degree(node))
            major_conn = G.degree(node)
            console.print(f"  📊 {node}: {full_conn} total connections ({major_conn} to major entities)")

        console.print(f"\n[yellow]Connections between major entities:[/yellow]")
        for edge in G.edges(data=True):
            entity_1, entity_2, data = edge
            weight = data.get('weight', 1)
            console.print(f"  {entity_1} ↔ {entity_2} (co-occurrences: {weight})")
    else:
        console.print(f"\n[red]No major entities found for visualization.[/red]")
        console.print(f"[yellow]Try lowering the minimum mentions threshold.[/yellow]")