import matplotlib.pyplot as plt
import numpy as np

# Data for the plots
x_values = [1, 2, 3]
local_values = [0.914,0.914,0.914]#0.898,0.898,0.898 #0.821,0.821,0.821 #0.914,0.914,0.914
args_values = [0.911, 0.938, 0.938]#0.748, 0.798, 0.804#0.620, 0.677, 0.711#0.911, 0.938, 0.938
return_values = [0.817, 0.868, 0.909]#0.556, 0.719, 0.742#0.753, 0.786, 0.779#0.817, 0.868, 0.909

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
#0.898,0.898,0.898 #0.821,0.821,0.821 #0.914,0.914,0.914
#0.748, 0.798, 0.804#0.620, 0.677, 0.711#0.911, 0.938, 0.938
#0.556, 0.719, 0.742#0.753, 0.786, 0.779#0.817, 0.868, 0.909