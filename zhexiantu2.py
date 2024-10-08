import matplotlib.pyplot as plt
import numpy as np

# Data for the plots
x_values = [1, 2, 3]
local_values = [0.703,0.703,0.703]#0.880,0.880,0.880 #0.678,0.678,0.678 #0.703,0.703,0.703
args_values = [0.392, 0.487, 0.507]#0.694, 0.815, 0.816#0.339, 0.511, 0.544#0.392, 0.487, 0.507
return_values = [0.256, 0.361, 0.349]#0.742, 0.834, 0.848#0.561, 0.697, 0.705#0.256, 0.361, 0.349

# Setting up the figure
plt.figure(figsize=(10, 6))

# Plotting data
plt.plot(x_values, local_values, label='Local', marker='o', linestyle='-', linewidth=2, color='blue')
plt.plot(x_values, args_values, label='Args', marker='o', linestyle='-', linewidth=2, color='orange')
plt.plot(x_values, return_values, label='Return', marker='o', linestyle='-', linewidth=2, color='green')

# Customize y-axis
y_min = min(min(local_values), min(args_values), min(return_values)) - 0.05
y_max = max(max(local_values), max(args_values), max(return_values)) + 0.05
plt.ylim(y_min, y_max)
plt.yticks(np.arange(np.floor(y_min*10)/10, np.ceil(y_max*10)/10, 0.05), ['{:.2f}'.format(x) for x in np.arange(np.floor(y_min*10)/10, np.ceil(y_max*10)/10, 0.05)])

# Customizing other elements of the plot
plt.xlabel('Reasoning Step')
plt.ylabel('Exact Match')
plt.xticks([1, 2, 3], ['Step 1', 'Step 2', 'Step 3'])

plt.legend()
plt.grid(True)
plt.show()

#0.880,0.880,0.880 #0.678,0.678，0.678 #0.703，0.703，0.703
#0.694, 0.815, 0.816#0.339, 0.511, 0.544#0.392, 0.487, 0.507
#0.742, 0.834, 0.848#0.561, 0.697, 0.705#0.256, 0.361, 0.349