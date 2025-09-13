# 🏛️ Nexus: AI-Powered Historical Network Analysis

> **Transforming historical texts into interactive relationship networks through advanced AI and natural language processing**

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Powered by spaCy](https://img.shields.io/badge/powered%20by-spaCy-09a3d5.svg)](https://spacy.io)
[![AI Enhanced](https://img.shields.io/badge/AI-Enhanced-purple.svg)](https://github.com/explosion/spaCy)

## 🎯 Purpose

Nexus is a specialized tool designed for **historians, researchers, and digital humanities scholars** who need to analyze complex relationships within historical texts. By leveraging state-of-the-art AI language models and network analysis techniques, Nexus automatically extracts character relationships and visualizes them as interactive, publication-ready networks.

### Why Nexus?

- **🤖 AI-Driven Analysis**: Utilizes transformer-based language models for accurate entity recognition
- **📊 Interactive Visualizations**: Creates stunning, glass-morphism styled network graphs
- **🎨 Publication Ready**: Generates high-quality HTML visualizations perfect for academic presentations
- **⚡ Rapid Insights**: Transform hours of manual analysis into minutes of automated processing
- **🔄 Continuously Evolving**: Regular updates incorporating the latest advances in NLP and visualization

## ✨ Features

### Core Capabilities
- **Advanced Entity Recognition**: Powered by spaCy's transformer models for precise character identification
- **Relationship Mapping**: Automatically detects co-occurrences and relationship strengths
- **Interactive Networks**: Beautiful, responsive visualizations with hover details and zoom capabilities
- **Statistical Analysis**: Comprehensive network metrics including centrality measures and connectivity patterns
- **Flexible Input**: Works with any plain text document (novels, historical documents, manuscripts)

### AI-Enhanced Visualizations
All visualizations in Nexus have been crafted using **Large Language Models (LLMs)** to ensure:
- **Optimal Visual Design**: AI-optimized layouts for maximum clarity and aesthetic appeal
- **Intelligent Color Schemes**: Contextually appropriate glass-morphism effects
- **Smart Node Sizing**: Proportional representation based on character importance
- **Enhanced Interactivity**: LLM-designed hover states and navigation patterns

## 🚀 Quick Start

### Prerequisites
- Python 3.12 or higher
- 4GB+ RAM (for transformer models)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/[your-username]/nexus.git
   cd nexus
   ```

2. **Set up the environment**
   ```bash
   # Using uv (recommended)
   uv sync
   
   # Or using pip
   pip install -r requirements.txt
   ```

3. **Download language models**
   ```bash
   python -m spacy download en_core_web_trf
   ```

### Basic Usage

```bash
# Analyze a text file with default settings
python main.py --file data/raw/your_document.txt

# Customize minimum mentions threshold
python main.py --file data/raw/your_document.txt --min-mentions 5

# Example with provided sample
python main.py --file data/raw/caesar.txt --min-mentions 3
```

## 📁 Project Structure

```
nexus/
├── src/                    # Core analysis modules
│   ├── entity_extractor.py    # AI-powered entity recognition
│   ├── graph_builder.py       # Network construction algorithms
│   ├── graph_visualizer.py    # LLM-enhanced visualization engine
│   ├── text_processor.py      # Advanced NLP preprocessing
│   └── cli.py                 # Command-line interface
├── data/
│   ├── raw/               # Input text files
│   └── processed/         # Generated visualizations
├── main.py                # Primary execution script
└── pyproject.toml         # Project configuration
```

## 🎨 Visualization Gallery

Nexus generates **interactive HTML visualizations** featuring:

- **Glass Morphism Design**: Modern, translucent aesthetic with depth and sophistication
- **Responsive Layouts**: Optimized for both desktop analysis and presentation displays
- **Rich Interactions**: Hover for detailed character information, zoom for focused analysis
- **Professional Styling**: Publication-ready graphics suitable for academic papers and presentations

*All visual designs have been enhanced through iterative collaboration with Large Language Models to ensure optimal user experience and aesthetic appeal.*

## 🔬 For Historians & Researchers

### Use Cases
- **Literary Analysis**: Map character relationships in historical novels and texts
- **Historical Documents**: Analyze correspondence networks in archival materials
- **Biographical Research**: Visualize social connections in historical biographies
- **Comparative Studies**: Compare relationship patterns across different time periods or authors
- **Digital Humanities**: Create interactive exhibits and educational materials

### Academic Integration
- Export visualizations for inclusion in papers and presentations
- Generate network statistics for quantitative analysis
- Create reproducible research workflows
- Support for various text formats and languages

## 🛠️ Advanced Configuration

### Command Line Options
```bash
python main.py [OPTIONS]

Options:
  --file TEXT          Path to input text file [default: data/raw/caesar.txt]
  --min-mentions INT   Minimum mentions for visualization [default: 3]
  --help              Show this message and exit
```

### Customization
The modular architecture allows for easy customization:
- Modify entity recognition parameters in `entity_extractor.py`
- Adjust visualization styles in `graph_visualizer.py`
- Extend analysis metrics in `graph_builder.py`

## 🔄 Continuous Evolution

Nexus is actively maintained and regularly updated with:
- **Latest NLP Models**: Integration of cutting-edge language models as they become available
- **Enhanced Visualizations**: Continuous improvement of visual designs through AI collaboration
- **Performance Optimizations**: Regular updates for faster processing and better scalability
- **Feature Expansions**: New analysis capabilities based on user feedback and research needs

## 📊 Technical Specifications

### Dependencies
- **spaCy 3.7+**: Advanced NLP and entity recognition
- **NetworkX 3.5+**: Graph analysis and algorithms
- **Plotly 5.17+**: Interactive visualization framework
- **NumPy & Pandas**: Data processing and analysis
- **Rich**: Enhanced command-line interface

### Performance
- Processes typical historical documents (50-200 pages) in under 2 minutes
- Scales efficiently with document size through optimized algorithms
- Memory usage optimized for standard research workstations

## 🤝 Contributing

We welcome contributions from the digital humanities community! Areas of particular interest:
- Historical text preprocessing improvements
- Additional visualization styles and themes
- Support for non-English historical documents
- Integration with digital archives and databases

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **spaCy Team**: For providing world-class NLP tools
- **Plotly Community**: For powerful visualization capabilities
- **Digital Humanities Community**: For inspiration and feedback
- **AI Research Community**: For advancing the language models that power our analysis

---

**Built for historians, by historians** • *Powered by AI, designed for discovery*

> "Every historical text contains hidden networks waiting to be discovered. Nexus makes that discovery possible." 

*Ready to explore the hidden connections in your historical texts? [Get started](#-quick-start) today.*