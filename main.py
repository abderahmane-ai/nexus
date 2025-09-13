import warnings
import argparse
from src.file_reader import read_file
from src.text_processor import process_text
from src.entity_extractor import get_unique_entities
from src.graph_builder import build_graph_from_sentences
from src.cli import print_sentences, print_graph_summary
from src.graph_visualizer import create_interactive_graph, save_interactive_html, show_graph_stats


# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

def main() -> None:
    parser = argparse.ArgumentParser(description='Analyze character networks in text documents')
    parser.add_argument('--min-mentions', type=int, default=3, 
                       help='Minimum mentions required for a character to appear in visualization (default: 3)')
    parser.add_argument('--file', type=str, default="data/raw/caesar.txt",
                       help='Path to the text file to analyze (default: data/raw/euthyphro.txt)')
    
    args = parser.parse_args()
    min_mentions = args.min_mentions
    file_path: str = args.file
    document_text: str = read_file(file_path)

    # Process the text with spaCy
    doc = process_text(document_text)
    sentences = list(doc.sents)  # Convert generator to list for multiple uses

    # Print first few sentences
    print_sentences(sentences)

    # Build graph from sentences (only major entities will be visualized)
    G = build_graph_from_sentences(sentences, min_mentions=min_mentions)

    # Print graph summary
    print_graph_summary(G, sentences)
    
    # Create interactive visualizations
    print("\n🎨 Creating visualization...")
    show_graph_stats(G)
    
    # Create interactive graph
    fig = create_interactive_graph(G, "Major Characters Network")
    save_interactive_html(fig, "data/processed/major_characters.html")

if __name__ == "__main__":
    main()