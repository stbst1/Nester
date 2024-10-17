import matplotlib.pyplot as plt
import numpy as np

# Data setup
groups = ['int', 'float', 'bool', 'str', 'bytes', 'list', 'tuple', 'dict', 'user']
locals_data = np.array([
    [0.898, 0.898, 0.898],
    [0.821, 0.821, 0.821],
    [0.914, 0.914, 0.914],
    [0.880, 0.880, 0.880],
    [0.678, 0.678, 0.678],
    [0.703, 0.703, 0.703],
    [0.393, 0.393, 0.393],
    [0.536, 0.536, 0.536],
    [0.876, 0.876, 0.876]
])
args_data = np.array([
    [0.748, 0.798, 0.804],
    [0.620, 0.677, 0.711],
    [0.911, 0.938, 0.938],
    [0.694, 0.815, 0.816],
    [0.339, 0.511, 0.544],
    [0.392, 0.487, 0.507],
    [0.273, 0.273, 0.455],
    [0.376, 0.574, 0.583],
    [0.744, 0.761, 0.775]
])
returns_data = np.array([
    [0.556, 0.719, 0.742],
    [0.753, 0.786, 0.779],
    [0.817, 0.868, 0.909],
    [0.742, 0.834, 0.848],
    [0.561, 0.697, 0.705],
    [0.256, 0.361, 0.349],
    [0.364, 0.364, 0.364],
    [0.519, 0.616, 0.643],
    [0.603, 0.657, 0.665]
])

# Calculate the mean for each set of local, args, and return for each group
mean_data = np.mean(np.array([locals_data, args_data, returns_data]), axis=0)

x = np.arange(len(groups))  # the label locations

# Plotting the data
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_facecolor('none')  
ax.grid(True, zorder=0, axis='y')

width = 0.25  # the width of the bars

# Colors for each category
colors = ['#CCCCCC', '#7CCD7C', '#EEA2AD']

# Border thickness
border_thickness = 2  # Adjust this value for different border thicknesses

bars1 = ax.bar(x - width, mean_data[:, 0], width, label='Step 1', color=colors[0], edgecolor='black', zorder=border_thickness)
bars2 = ax.bar(x, mean_data[:, 1], width, label='Step 2', color=colors[1], edgecolor='black', zorder=border_thickness)
bars3 = ax.bar(x + width, mean_data[:, 2], width, label='Step 3', color=colors[2], edgecolor='black', zorder=border_thickness)

# Annotating each bar
for rects in [bars1, bars2, bars3]:
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}', xy=(rect.get_x() + rect.get_width() / 2, height), xytext=(0, 5),  
                    textcoords="offset points", ha='center', va='bottom', fontsize=18, rotation=90)

# Set labels and ticks
ax.set_xticks(x)
ax.set_xticklabels(groups)
ax.set_ylabel('Average Scores', fontsize=27)
ax.tick_params(axis='y', labelsize=27)
ax.tick_params(axis='x', labelsize=27)
ax.legend(loc='center right', bbox_to_anchor=(0.8, 0.84), fontsize=24, frameon=True)

# Adjust x-axis limits to fit the bars snugly
ax.set_xlim(-0.5, len(groups) - 0.5)  # Adjust these values to reduce the gap on the sides

# Adjust y-axis limit to fit annotations
ax.set_ylim(0, 1.05)  # Setting a fixed upper limit based on the observed data values

fig.tight_layout()

plt.show()





