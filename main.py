import warnings
import argparse
import logging
from src import FileReader, TextProcessor, EntityExtractor, GraphBuilder, CLI, GraphVisualizer


# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


class TextNetworkAnalyzer:
    """Main application class for analyzing character networks in text documents."""
    
    def __init__(self):
        """Initialize the analyzer with all necessary components."""
        self.file_reader = FileReader()
        self.text_processor = TextProcessor()
        self.entity_extractor = EntityExtractor()
        self.graph_builder = GraphBuilder(self.entity_extractor, use_sentiment=True)
        self.cli = CLI(self.entity_extractor)
        self.graph_visualizer = GraphVisualizer()
    
    def analyze_document(self, file_path: str, min_mentions: int = 3, verbose: bool = True) -> None:
        """
        Analyze a document and create network visualization.
        
        Args:
            file_path: Path to the text file to analyze
            min_mentions: Minimum mentions required for inclusion in visualization
            verbose: Whether to enable verbose output
        """
        try:
            # Read the document
            self.cli.print_info(f"Reading document: {file_path}")
            document_text = self.file_reader.read_file(file_path)
            
            # Process the text with spaCy
            self.cli.print_info("Processing text with spaCy...")
            sentences = self.text_processor.get_sentences(document_text)
            
            # Print first few sentences
            self.cli.print_sentences(sentences)
            
            # Build graph from sentences
            self.cli.print_info("Building character network...")
            G = self.graph_builder.build_graph_from_sentences(
                sentences, 
                min_mentions=min_mentions, 
                verbose=verbose
            )
            
            # Print graph summary
            self.cli.print_graph_summary(G, sentences)
            
            # Create interactive visualizations
            if G.number_of_nodes() > 0:
                self.cli.print_info("Creating visualization...")
                self.graph_visualizer.show_graph_stats(G)
                
                # Create interactive graph
                fig = self.graph_visualizer.create_interactive_graph(G, "Major Characters Network")
                self.graph_visualizer.save_interactive_html(fig, "data/processed/major_characters.html")
                self.cli.print_success("Analysis complete! Check data/processed/major_characters.html")
            else:
                self.cli.print_error("No entities found for visualization. Try lowering --min-mentions.")
                
        except FileNotFoundError as e:
            self.cli.print_error(f"File not found: {e}")
        except Exception as e:
            self.cli.print_error(f"An error occurred: {e}")
            if verbose:
                raise


def main() -> None:
    """Main entry point for the application."""
    parser = argparse.ArgumentParser(description='Analyze character networks in text documents')
    parser.add_argument('--min-mentions', type=int, default=3, 
                       help='Minimum mentions required for a character to appear in visualization (default: 3)')
    parser.add_argument('--file', type=str, default="data/raw/data.txt",
                       help='Path to the text file to analyze (default: data/raw/data.txt)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output for debugging')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(level=log_level, format='%(name)s - %(levelname)s - %(message)s')
    
    # Create and run analyzer
    analyzer = TextNetworkAnalyzer()
    analyzer.analyze_document(
        file_path=args.file,
        min_mentions=args.min_mentions,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()