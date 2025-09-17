from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
import networkx as nx
from typing import List
import spacy
from ..core.entity_extractor import EntityExtractor


class CLI:
    """Handles command-line interface and rich text output."""
    
    def __init__(self, entity_extractor: EntityExtractor = None):
        """
        Initialize the CLI handler.
        
        Args:
            entity_extractor: EntityExtractor instance for entity analysis
        """
        self.console = Console()
        self.entity_extractor = entity_extractor or EntityExtractor()
    
    def print_sentences(self, sentences: List[spacy.tokens.Span], limit: int = 3) -> None:
        """Print the first few sentences with rich formatting."""
        self.console.print(Panel("📄 Document Preview", style="bold blue"))
        for i, sentence in enumerate(sentences[:limit], 1):
            self.console.print(f"[cyan]{i}.[/cyan] {sentence.text[:100]}{'...' if len(sentence.text) > 100 else ''}")
    
    def print_graph_summary(self, G: nx.Graph, sentences: List[spacy.tokens.Span]) -> None:
        """Print a concise summary of the graph using rich."""
        # Calculate total entities from sentences
        all_entities = []
        for sentence in sentences:
            unique_entities = self.entity_extractor.get_unique_entities(sentence)
            all_entities.extend(unique_entities)
        
        self.console.print(Panel("📊 Network Summary", style="bold magenta"))
        self.console.print(f"[green]Nodes:[/green] {G.number_of_nodes()} | [green]Connections:[/green] {G.number_of_edges()} | [green]Density:[/green] {nx.density(G):.3f}")

        if G.number_of_nodes() > 0:
            # Show top 5 most connected entities
            degrees = dict(G.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:5]
            self.console.print(f"\n[yellow]Most connected:[/yellow] " + " | ".join([f"{node}: {degree}" for node, degree in top_nodes]))
            
            # Show sentiment analysis if available
            sentiment_edges = [(u, v, data) for u, v, data in G.edges(data=True) if 'sentiment' in data]
            if sentiment_edges:
                avg_sentiment = sum(data['sentiment'] for _, _, data in sentiment_edges) / len(sentiment_edges)
                sentiment_color = "green" if avg_sentiment > 0.1 else "red" if avg_sentiment < -0.1 else "yellow"
                self.console.print(f"[{sentiment_color}]Average sentiment:[/{sentiment_color}] {avg_sentiment:.3f}")
                
                # Show most positive and negative relationships
                sorted_by_sentiment = sorted(sentiment_edges, key=lambda x: x[2]['sentiment'])
                if len(sorted_by_sentiment) >= 2:
                    most_negative = sorted_by_sentiment[0]
                    most_positive = sorted_by_sentiment[-1]
                    self.console.print(f"[red]Most negative:[/red] {most_negative[0]} ↔ {most_negative[1]} ({most_negative[2]['sentiment']:.3f})")
                    self.console.print(f"[green]Most positive:[/green] {most_positive[0]} ↔ {most_positive[1]} ({most_positive[2]['sentiment']:.3f})")
        else:
            self.console.print(f"\n[red]No major entities found. Try lowering --min-mentions.[/red]")
    
    def print_entity_table(self, entity_stats: dict, limit: int = 10) -> None:
        """Print a table of entities and their occurrence counts."""
        table = Table(title="Entity Statistics")
        table.add_column("Entity", style="cyan", no_wrap=True)
        table.add_column("Mentions", style="magenta", justify="right")
        table.add_column("Status", style="green")
        
        sorted_entities = sorted(entity_stats.items(), key=lambda x: x[1], reverse=True)
        
        for entity, count in sorted_entities[:limit]:
            status = "Major" if count >= 3 else "Minor"  # Default threshold
            table.add_row(entity, str(count), status)
        
        self.console.print(table)
    
    def print_success(self, message: str) -> None:
        """Print a success message."""
        self.console.print(f"✅ {message}", style="bold green")
    
    def print_error(self, message: str) -> None:
        """Print an error message."""
        self.console.print(f"❌ {message}", style="bold red")
    
    def print_info(self, message: str) -> None:
        """Print an info message."""
        self.console.print(f"ℹ️  {message}", style="bold blue")