import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional
import numpy as np


class GraphVisualizer:
    """Handles visualization of network graphs using Plotly."""
    
    def __init__(self, layout_algorithm: str = "spring", iterations: int = 50):
        """
        Initialize the graph visualizer.
        
        Args:
            layout_algorithm: Layout algorithm to use for node positioning
            iterations: Number of iterations for layout algorithm
        """
        self.layout_algorithm = layout_algorithm
        self.iterations = iterations
        self.layout_params = {"k": 3, "iterations": iterations}
    
    def create_interactive_graph(self, G: nx.Graph, title: str = "Major Characters Network") -> go.Figure:
        """Create an interactive Plotly visualization of the NetworkX graph."""
        
        # Use specified layout for node positioning
        if self.layout_algorithm == "spring":
            pos = nx.spring_layout(G, **self.layout_params)
        elif self.layout_algorithm == "circular":
            pos = nx.circular_layout(G)
        elif self.layout_algorithm == "random":
            pos = nx.random_layout(G)
        else:
            pos = nx.spring_layout(G, **self.layout_params)
        
        # Extract edges with sentiment information
        edge_x = []
        edge_y = []
        edge_weights = []
        edge_sentiments = []
        edge_colors = []
        edge_widths = []
        edge_hover_text = []
        
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            
            weight = edge[2]['weight']
            sentiment = edge[2].get('sentiment', 0.0)
            edge_weights.append(weight)
            edge_sentiments.append(sentiment)
            
            # Color edges based on sentiment
            if sentiment > 0.2:
                color = f'rgba(100, 255, 150, {0.6 + sentiment * 0.3})'  # Green for positive
            elif sentiment < -0.2:
                color = f'rgba(255, 100, 120, {0.6 + abs(sentiment) * 0.3})'  # Red for negative
            else:
                color = 'rgba(180, 200, 255, 0.4)'  # Blue-white for neutral
            
            edge_colors.extend([color, color, None])
            
            # Width based on connection strength
            width = max(1.5, min(6, weight * 0.5))
            edge_widths.extend([width, width, None])
            
            # Hover text with sentiment info
            hover_text = f"{edge[0]} ↔ {edge[1]}<br>Connections: {weight}<br>Sentiment: {sentiment:.3f}"
            edge_hover_text.extend([hover_text, hover_text, None])
        
        # Create multiple edge traces for different sentiment categories
        edge_traces = []
        
        if edge_sentiments:
            # Group edges by sentiment category
            positive_edges = {'x': [], 'y': [], 'hover': []}
            negative_edges = {'x': [], 'y': [], 'hover': []}
            neutral_edges = {'x': [], 'y': [], 'hover': []}
            
            i = 0
            for edge in G.edges(data=True):
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                sentiment = edge[2].get('sentiment', 0.0)
                weight = edge[2]['weight']
                hover_text = f"{edge[0]} ↔ {edge[1]}<br>Connections: {weight}<br>Sentiment: {sentiment:.3f}"
                
                if sentiment > 0.2:
                    positive_edges['x'].extend([x0, x1, None])
                    positive_edges['y'].extend([y0, y1, None])
                    positive_edges['hover'].extend([hover_text, hover_text, None])
                elif sentiment < -0.2:
                    negative_edges['x'].extend([x0, x1, None])
                    negative_edges['y'].extend([y0, y1, None])
                    negative_edges['hover'].extend([hover_text, hover_text, None])
                else:
                    neutral_edges['x'].extend([x0, x1, None])
                    neutral_edges['y'].extend([y0, y1, None])
                    neutral_edges['hover'].extend([hover_text, hover_text, None])
            
            # Create separate traces for each sentiment category
            if positive_edges['x']:
                edge_traces.append(go.Scatter(
                    x=positive_edges['x'], y=positive_edges['y'],
                    line=dict(width=2.5, color='rgba(100, 255, 150, 0.7)'),
                    hoverinfo='text', hovertext=positive_edges['hover'],
                    mode='lines', name='Positive', showlegend=False
                ))
            
            if negative_edges['x']:
                edge_traces.append(go.Scatter(
                    x=negative_edges['x'], y=negative_edges['y'],
                    line=dict(width=2.5, color='rgba(255, 100, 120, 0.7)'),
                    hoverinfo='text', hovertext=negative_edges['hover'],
                    mode='lines', name='Negative', showlegend=False
                ))
            
            if neutral_edges['x']:
                edge_traces.append(go.Scatter(
                    x=neutral_edges['x'], y=neutral_edges['y'],
                    line=dict(width=1.5, color='rgba(180, 200, 255, 0.4)'),
                    hoverinfo='text', hovertext=neutral_edges['hover'],
                    mode='lines', name='Neutral', showlegend=False
                ))
        else:
            # Fallback to original single trace if no sentiment data
            edge_traces.append(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=1.5, color='rgba(180, 200, 255, 0.4)'),
                hoverinfo='none', mode='lines'
            ))
        
        # Extract nodes
        node_x = []
        node_y = []
        node_text = []
        node_info = []
        full_connections = []
        
        # First pass: collect all full connection counts to calculate scaling
        for node in G.nodes():
            full_conn_count = G.nodes[node].get('full_connections', G.degree(node))
            full_connections.append(full_conn_count)
        
        # Calculate size scaling based on the actual range of full connections
        min_connections = min(full_connections) if full_connections else 0
        max_connections = max(full_connections) if full_connections else 1
        connections_range = max_connections - min_connections if max_connections > min_connections else 1
        
        # Define size range: smallest nodes = 15, largest nodes = 100
        min_size, max_size = 15, 100
        
        node_sizes = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            
            # Get full connection count (including connections with minor entities)
            full_conn_count = G.nodes[node].get('full_connections', G.degree(node))
            major_conn_count = G.degree(node)  # Connections only to other major entities
            
            # Calculate node size based on full connections with proper scaling
            if connections_range > 0:
                normalized_connections = (full_conn_count - min_connections) / connections_range
                size = min_size + (max_size - min_size) * normalized_connections
            else:
                size = (min_size + max_size) / 2  # Use middle size if all nodes have same connections
            
            node_sizes.append(size)
            
            # Create hover info showing connections and sentiment
            adjacencies = list(G.neighbors(node))
            
            # Calculate average sentiment for this node's relationships
            node_sentiments = []
            for neighbor in adjacencies:
                edge_data = G[node][neighbor]
                if 'sentiment' in edge_data:
                    node_sentiments.append(edge_data['sentiment'])
            
            avg_sentiment = sum(node_sentiments) / len(node_sentiments) if node_sentiments else 0.0
            sentiment_emoji = "😊" if avg_sentiment > 0.1 else "😔" if avg_sentiment < -0.1 else "😐"
            
            node_info.append(f'{node} {sentiment_emoji}<br>Total connections: {full_conn_count}<br>' +
                            f'Major entity connections: {major_conn_count}<br>' +
                            f'Average sentiment: {avg_sentiment:.3f}<br>' +
                            f'Connected to: {", ".join(adjacencies[:5])}' +
                            ('...' if len(adjacencies) > 5 else ''))
        
        # Create sophisticated glass effect colors with depth
        glass_colors = []
        for i, size in enumerate(node_sizes):
            # Create a gradient from cool blue to warm purple based on connections
            normalized_size = (size - min_size) / (max_size - min_size) if max_size > min_size else 0.5
            
            # Multi-layered glass colors inspired by iOS design
            if normalized_size < 0.25:
                # Ice glass - very light blue with high transparency
                glass_colors.append(f'rgba(135, 195, 255, 0.72)')
            elif normalized_size < 0.5:
                # Cool glass - blue-cyan with medium transparency
                glass_colors.append(f'rgba(120, 180, 255, 0.78)')
            elif normalized_size < 0.75:
                # Warm glass - purple-blue with stronger presence
                glass_colors.append(f'rgba(150, 120, 255, 0.82)')
            else:
                # Premium glass - rich purple-pink with depth
                glass_colors.append(f'rgba(200, 120, 255, 0.88)')
        
        # Create node trace with glass effect
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            hoverinfo='text',
            text=node_text,
            textposition="middle center",
            hovertext=node_info,
            marker=dict(
                size=node_sizes,
                color=glass_colors,
                # Add multiple border layers for glass depth effect
                line=dict(
                    width=3,
                    color='rgba(255, 255, 255, 0.8)'  # Bright white border for glass rim
                ),
                # Add shadow effect with opacity
                opacity=0.9
            )
        )
        
        # Create the figure with glass-themed background
        fig = go.Figure(data=edge_traces + [node_trace],
                       layout=go.Layout(
                            title=dict(
                                text=title, 
                                font=dict(size=18, color='rgba(100, 120, 180, 0.9)', family="SF Pro Display, -apple-system, sans-serif")
                            ),
                            showlegend=False,
                            hovermode='closest',
                            margin=dict(b=20,l=5,r=5,t=40),
                            annotations=[ dict(
                                text="✨ Glass-effect visualization • Sentiment-aware connections • Only major characters shown",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002,
                                xanchor='left', yanchor='bottom',
                                font=dict(color='rgba(120, 140, 200, 0.7)', size=11, family="SF Pro Display, -apple-system, sans-serif")
                            )],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            # iOS-style glass background
                            plot_bgcolor='rgba(245, 248, 255, 0.95)',
                            paper_bgcolor='rgba(248, 250, 255, 0.95)'
                        ))
        
        return fig

    
    def save_interactive_html(self, fig: go.Figure, filename: str = "graph_visualization.html") -> None:
        """Save the interactive graph as an HTML file with glass effect styling."""
        
        # Custom CSS for advanced glass morphism effects
        glass_css = """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=SF+Pro+Display:wght@300;400;500;600&display=swap');
        
        body {
            background: 
                radial-gradient(circle at 20% 80%, rgba(120, 180, 255, 0.15) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(200, 120, 255, 0.15) 0%, transparent 50%),
                linear-gradient(135deg, 
                    rgba(240, 245, 255, 0.95) 0%, 
                    rgba(250, 250, 255, 0.98) 50%,
                    rgba(245, 248, 255, 0.95) 100%);
            backdrop-filter: blur(30px);
            -webkit-backdrop-filter: blur(30px);
            font-family: 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
            min-height: 100vh;
            margin: 0;
            padding: 20px;
        }
        
        .plotly-graph-div {
            border-radius: 24px;
            backdrop-filter: blur(60px);
            -webkit-backdrop-filter: blur(60px);
            box-shadow: 
                0 20px 60px rgba(120, 180, 255, 0.12),
                0 8px 32px rgba(150, 120, 255, 0.08),
                0 2px 8px rgba(200, 120, 255, 0.05),
                inset 0 1px 0 rgba(255, 255, 255, 0.4),
                inset 0 -1px 0 rgba(255, 255, 255, 0.1);
            border: 1.5px solid rgba(255, 255, 255, 0.25);
            background: 
                linear-gradient(135deg, 
                    rgba(255, 255, 255, 0.15) 0%,
                    rgba(255, 255, 255, 0.05) 100%);
            position: relative;
            overflow: hidden;
        }
        
        .plotly-graph-div::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 1px;
            background: linear-gradient(90deg, 
                transparent 0%,
                rgba(255, 255, 255, 0.6) 50%,
                transparent 100%);
            z-index: 1;
        }
        
        /* Enhanced glass effect for controls */
        .plotly .modebar {
            background: rgba(255, 255, 255, 0.85) !important;
            backdrop-filter: blur(20px) !important;
            -webkit-backdrop-filter: blur(20px) !important;
            border-radius: 16px !important;
            border: 1px solid rgba(255, 255, 255, 0.4) !important;
            box-shadow: 
                0 8px 24px rgba(120, 180, 255, 0.1),
                inset 0 1px 0 rgba(255, 255, 255, 0.5) !important;
            padding: 4px !important;
        }
        
        .plotly .modebar-btn {
            background: rgba(255, 255, 255, 0.7) !important;
            border-radius: 10px !important;
            margin: 2px !important;
            transition: all 0.4s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
            border: 1px solid rgba(255, 255, 255, 0.3) !important;
        }
        
        .plotly .modebar-btn:hover {
            background: rgba(120, 180, 255, 0.85) !important;
            transform: translateY(-2px) scale(1.05) !important;
            box-shadow: 
                0 8px 20px rgba(120, 180, 255, 0.25),
                0 4px 12px rgba(120, 180, 255, 0.15) !important;
            border: 1px solid rgba(120, 180, 255, 0.4) !important;
        }
        
        /* Subtle animation for the entire container */
        .plotly-graph-div {
            animation: glassFloat 6s ease-in-out infinite;
        }
        
        @keyframes glassFloat {
            0%, 100% { 
                transform: translateY(0px) rotate(0deg);
                box-shadow: 
                    0 20px 60px rgba(120, 180, 255, 0.12),
                    0 8px 32px rgba(150, 120, 255, 0.08);
            }
            50% { 
                transform: translateY(-2px) rotate(0.1deg);
                box-shadow: 
                    0 25px 70px rgba(120, 180, 255, 0.15),
                    0 12px 40px rgba(150, 120, 255, 0.1);
            }
        }
        
        /* Glass reflection effect */
        .plotly-graph-div::after {
            content: '';
            position: absolute;
            top: 10%;
            left: 10%;
            width: 30%;
            height: 30%;
            background: linear-gradient(135deg, 
                rgba(255, 255, 255, 0.3) 0%,
                rgba(255, 255, 255, 0.1) 50%,
                transparent 100%);
            border-radius: 50%;
            filter: blur(20px);
            pointer-events: none;
            z-index: 1;
        }
    </style>
        """
        
        # Write HTML with custom styling
        html_string = fig.to_html(include_plotlyjs=True)
        
        # Insert custom CSS before closing head tag
        html_with_glass = html_string.replace('</head>', f'{glass_css}</head>')
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_with_glass)
        
        print(f"✨ Saved: {filename}")
    
    def show_graph_stats(self, G: nx.Graph) -> None:
        """Display concise graph statistics."""
        if G.number_of_nodes() > 0:
            degrees = dict(G.degree())
            top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:3]
            print(f"🔗 Top connections: " + " | ".join([f"{node}: {degree}" for node, degree in top_nodes]))
    
    def set_layout_algorithm(self, algorithm: str, **params) -> None:
        """
        Set the layout algorithm and parameters.
        
        Args:
            algorithm: Layout algorithm name ('spring', 'circular', 'random')
            **params: Additional parameters for the layout algorithm
        """
        self.layout_algorithm = algorithm
        if algorithm == "spring":
            self.layout_params.update(params)
        
    def get_available_layouts(self) -> List[str]:
        """Get list of available layout algorithms."""
        return ["spring", "circular", "random"]