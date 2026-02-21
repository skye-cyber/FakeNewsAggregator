import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np

# Create the figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Define the tasks with their start times and durations
tasks = [
    {'name': '1. Problem Definition & Literature Review', 'day': 0, 'duration': 1},
    {'name': '2. Data Collection & Preprocessing', 'day': 0, 'duration': 1},
    {'name': '3. Model Architecture Design', 'day': 1, 'duration': 1},
    {'name': '4. Initial Implementation', 'day': 1, 'duration': 1},
    {'name': '5. Model Training, Validation, and Fine-tuning', 'day': 2, 'duration': 1},
    {'name': '6. Final Testing and Documentation', 'day': 2, 'duration': 1}
]

# Base date (you can set this to any starting date)
base_date = datetime(2024, 1, 1)

# Colors for the bars
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']

# Create the Gantt chart
for i, task in enumerate(tasks):
    start_date = base_date + timedelta(days=task['day'])
    end_date = start_date + timedelta(days=task['duration'])

    # Create the bar for this task
    ax.barh(task['name'], task['duration'], left=task['day'],
            color=colors[i % len(colors)], edgecolor='black', alpha=0.7)

    # Add task label in the middle of the bar
    ax.text(task['day'] + task['duration']/2, i, task['name'],
            ha='center', va='center', fontweight='bold', fontsize=9)

# Customize the chart
ax.set_xlabel('Days', fontweight='bold', fontsize=12)
ax.set_ylabel('Tasks', fontweight='bold', fontsize=12)
ax.set_title('Project Timeline - 3 Day Gantt Chart', fontweight='bold', fontsize=14)

# Set x-axis limits and ticks
ax.set_xlim(-0.5, 3.5)
ax.set_xticks([0, 1, 2, 3])
ax.set_xticklabels(['Day 1', 'Day 2', 'Day 3', ''])

# Remove y-axis labels since we're putting labels in the bars
ax.set_yticks([])

# Add grid for better readability
ax.grid(True, axis='x', alpha=0.3)
ax.set_axisbelow(True)

# Adjust layout and display
plt.tight_layout()
plt.show()

# Optional: Save the chart
# plt.savefig('gantt_chart.png', dpi=300, bbox_inches='tight')
