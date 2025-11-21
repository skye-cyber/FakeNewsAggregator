import matplotlib.pyplot as plt
# Evolution of architectures over time
fig, ax = plt.subplots(figsize=(14, 8))

# Timeline data
years = [2010, 2013, 2014, 2017, 2018]
architectures_timeline = ['Simple RNNs', 'LSTM Improvement', 'CNN for Text', 'CNN-LSTM Hybrid', 'Transformer Revolution']
performance_timeline = [60, 70, 75, 85, 95]

# Create timeline
ax.plot(years, performance_timeline, 'o-', linewidth=3, markersize=10,
        color='#2E86AB', markerfacecolor='#A23B72', markeredgecolor='white', markeredgewidth=2)

# Annotate each point
for i, (year, arch, perf) in enumerate(zip(years, architectures_timeline, performance_timeline)):
    ax.annotate(f'{arch}\n({perf}%)',
                (year, perf),
                textcoords="offset points",
                xytext=(0, 20 if i % 2 == 0 else -30),
                ha='center',
                fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7),
                arrowprops=dict(arrowstyle="->", color='black'))

ax.set_xlabel('Year', fontweight='bold', fontsize=12)
ax.set_ylabel('Typical Performance (%)', fontweight='bold', fontsize=12)
ax.set_title('Evolution of Text Classification Architectures', fontweight='bold', fontsize=14)
ax.grid(True, alpha=0.3)
ax.set_xlim(2009, 2019)
ax.set_ylim(50, 100)

plt.tight_layout()
plt.savefig('architecture_evolution.png', dpi=300, bbox_inches='tight')
plt.show()
