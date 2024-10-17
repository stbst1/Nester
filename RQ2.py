import matplotlib.pyplot as plt
import numpy as np

# Categories and accuracy data
categories = ['Optional', 'Union', 'Pattern', 'Mat', 'Defdict', 'Type', 'Deque', 'IO']
accuracy_NSTI = [0.58, 0.17, 0.99, 0.92, 0.64, 0.83, 0.9, 1]
accuracy_TypeGen = [0.12, 0.11, 0.69, 0.29, 0, 0.36, 0, 0.25]

# Define the colors
colors = ['#7CCD7C', '#EEA2AD']

# Bar width
bar_width = 0.35

# Index for the categories
index = np.arange(len(categories))

# Creating the bar plot
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_facecolor('none')  # Background transparency
ax.grid(True, zorder=0, axis='y')  # Display grid on y-axis behind bars

bars1 = ax.bar(index, accuracy_NSTI, bar_width, label='Nester', color=colors[0], edgecolor='black', zorder=3)
bars2 = ax.bar(index + bar_width, accuracy_TypeGen, bar_width, label='TypeGen', color=colors[1], edgecolor='black', zorder=3)

# Adding labels with adjustments
ax.set_ylabel('Precision', fontsize=27)
ax.tick_params(axis='y', labelsize=27)
ax.tick_params(axis='x', labelsize=27)
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(categories, fontsize=24)
ax.legend(loc='upper left', fontsize=24, frameon=True)

# Adding value labels to bars
def add_value_labels(bars, fontsize=18):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=fontsize)

add_value_labels(bars1)
add_value_labels(bars2)

# Adjust x-axis limits to fit the bars snugly
start = index[0] - 0.5 * bar_width - 0.2
end = index[-1] + 1.5 * bar_width + 0.2
ax.set_xlim(start, end)

# Display the plot
plt.show()
















#call iter gener
# 类别
#categories = ['typing.Optional', 'typing.Union', 'typing.Pattern', 'typing.Match',
#              'typing.defaultdict', 'typing.Type', 'typing.Deque', 'typing.IO']
