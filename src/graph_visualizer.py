import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple
import numpy as np

def create_interactive_graph(G: nx.Graph, title: str = "Major Characters Network") -> go.Figure:
    """Create an interactive Plotly visualization of the NetworkX graph."""
    
    # Use spring layout for better node positioning
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Extract edges
    edge_x = []
    edge_y = []
    edge_weights = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_weights.append(G[edge[0]][edge[1]]['weight'])
    
    # Create edge trace with glass-like connections
    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(
            width=1.5, 
            color='rgba(180, 200, 255, 0.4)'  # Soft blue-white for glass connections
        ),
        hoverinfo='none',
        mode='lines'
    )
    
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
        
        # Create hover info showing both full and major connections
        adjacencies = list(G.neighbors(node))
        node_info.append(f'{node}<br>Total connections: {full_conn_count}<br>' +
                        f'Major entity connections: {major_conn_count}<br>' +
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
    fig = go.Figure(data=[edge_trace, node_trace],
                   layout=go.Layout(
                        title=dict(
                            text=title, 
                            font=dict(size=18, color='rgba(100, 120, 180, 0.9)', family="SF Pro Display, -apple-system, sans-serif")
                        ),
                        showlegend=False,
                        hovermode='closest',
                        margin=dict(b=20,l=5,r=5,t=40),
                        annotations=[ dict(
                            text="✨ Glass-effect visualization • Only major characters shown",
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



def save_interactive_html(fig: go.Figure, filename: str = "graph_visualization.html") -> None:
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

def show_graph_stats(G: nx.Graph) -> None:
    """Display concise graph statistics."""
    if G.number_of_nodes() > 0:
        degrees = dict(G.degree())
        top_nodes = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:3]
        print(f"🔗 Top connections: " + " | ".join([f"{node}: {degree}" for node, degree in top_nodes]))