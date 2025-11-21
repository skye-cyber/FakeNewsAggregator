import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

# Create the three-layer architecture diagram
fig, ax = plt.subplots(1, 1, figsize=(14, 10))
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Title
ax.text(5, 11.5, 'Three-Layer AI System Architecture',
        ha='center', va='center', fontsize=18, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8))

# Layer definitions with components
layers = [
    {
        'name': 'DATA LAYER',
        'y_pos': 9,
        'color': '#FF6B6B',
        'components': [
            'Data Collection\n(Web scraping, APIs, DB)',
            'Data Cleaning & Preprocessing',
            'Feature Engineering',
            'Data Pipeline Management',
            'Storage (SQL/NoSQL)'
        ]
    },
    {
        'name': 'MODEL LAYER',
        'y_pos': 5.5,
        'color': '#4ECDC4',
        'components': [
            'Deep Learning Architecture',
            'Model Training & Validation',
            'Hyperparameter Tuning',
            'Model Monitoring & Logging',
            'Performance Metrics'
        ]
    },
    {
        'name': 'APPLICATION LAYER',
        'y_pos': 2,
        'color': '#45B7D1',
        'components': [
            'Web Interface (Frontend)',
            'REST API Endpoints',
            'Real-time Inference',
            'Visualization Dashboard',
            'User Management'
        ]
    }
]

# Draw layers
for layer in layers:
    # Main layer box
    rect = FancyBboxPatch((1, layer['y_pos'] - 1), 8, 2,
                         boxstyle="round,pad=0.1", linewidth=3,
                         edgecolor='black', facecolor=layer['color'], alpha=0.8)
    ax.add_patch(rect)
    ax.text(5, layer['y_pos'], layer['name'], ha='center', va='center',
            fontweight='bold', fontsize=14, color='white')

    # Components
    component_width = 1.4
    component_height = 0.8
    spacing = 0.1
    total_width = len(layer['components']) * component_width + (len(layer['components']) - 1) * spacing
    start_x = (10 - total_width) / 2

    for i, component in enumerate(layer['components']):
        x_pos = start_x + i * (component_width + spacing)
        y_pos = layer['y_pos'] - 1.8

        comp_rect = FancyBboxPatch((x_pos, y_pos), component_width, component_height,
                                 boxstyle="round,pad=0.05", linewidth=1,
                                 edgecolor='black', facecolor='white', alpha=0.9)
        ax.add_patch(comp_rect)
        ax.text(x_pos + component_width/2, y_pos + component_height/2, component,
                ha='center', va='center', fontsize=8, fontweight='bold', wrap=True)

# Draw data flow arrows
arrow_props = dict(arrowstyle="->", color='black', lw=2, alpha=0.7)
ax.annotate("", xy=(5, 8.2), xytext=(5, 7.3), arrowprops=arrow_props)
ax.annotate("", xy=(5, 4.7), xytext=(5, 3.8), arrowprops=arrow_props)
ax.text(5.2, 6, "Processed Data", fontweight='bold', fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow'))
ax.text(5.2, 3.2, "Predictions", fontweight='bold', fontsize=10,
        bbox=dict(boxstyle="round,pad=0.2", facecolor='yellow'))

plt.tight_layout()
plt.savefig('three_layer_architecture.png', dpi=300, bbox_inches='tight')
plt.show()
