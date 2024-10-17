import matplotlib.pyplot as plt
import numpy as np

# Data for the plots
x_values = [1, 2, 3]
local_values = [0.876,0.876,0.876]#0.393,0.393,0.393 #0.536,0.536,0.536 #0.876,0.876,0.876
args_values = [0.744, 0.761, 0.775]#0.273,0.273,0.455#0.376, 0.574, 0.583 #0.744, 0.761, 0.775
return_values = [0.603,0.657,0.665]#0.364,0.364,0.364#0.519,0.616,0.643 #0.603,0.657,0.665

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

#0.393,0.393,0.393 #0.536,0.536,0.536 #0.876,0.876,0.876
#0.273,0.273,0.455#0.376, 0.574, 0.583 #0.744, 0.761, 0.775
#0.364,0.364,0.364#0.519,0.616,0.643 #0.603,0.657,0.665
