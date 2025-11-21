import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

# Set style
plt.style.use('seaborn-v0_8')
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle('Neural Network Architectures for Text Classification', fontsize=16, fontweight='bold')

# Data
architectures = ['RNN/LSTM', 'CNN', 'CNN-LSTM\n(Hybrid)', 'Transformer\n(BERT/GPT)']
performance = [75, 82, 88, 95]  # Typical accuracy scores
training_speed = [35, 85, 60, 20]  # Relative speed (higher = faster)
computational_cost = [60, 75, 70, 15]  # Resource requirements (higher = more expensive)
data_requirements = [50, 60, 65, 90]  # Data needed (higher = more data required)

# 1. Performance Comparison
bars1 = ax1.bar(architectures, performance, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
ax1.set_title('Model Performance (Accuracy %)', fontweight='bold', pad=20)
ax1.set_ylabel('Accuracy (%)')
ax1.set_ylim(0, 100)
# Add value labels on bars
for bar in bars1:
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height}%', ha='center', va='bottom', fontweight='bold')

# 2. Training Speed Comparison
bars2 = ax2.bar(architectures, training_speed, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
ax2.set_title('Training Speed (Relative)', fontweight='bold', pad=20)
ax2.set_ylabel('Speed Score (Higher = Faster)')
ax2.set_ylim(0, 100)
for bar in bars2:
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height}', ha='center', va='bottom', fontweight='bold')

# 3. Computational Requirements
bars3 = ax3.bar(architectures, computational_cost, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
ax3.set_title('Computational Cost (Relative)', fontweight='bold', pad=20)
ax3.set_ylabel('Cost Score (Higher = More Expensive)')
ax3.set_ylim(0, 100)
for bar in bars3:
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height}', ha='center', va='bottom', fontweight='bold')

# 4. Data Requirements
bars4 = ax4.bar(architectures, data_requirements, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
ax4.set_title('Data Requirements (Relative)', fontweight='bold', pad=20)
ax4.set_ylabel('Data Score (Higher = More Data Needed)')
ax4.set_ylim(0, 100)
for bar in bars4:
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
             f'{height}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('architecture_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
