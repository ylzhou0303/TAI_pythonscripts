import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Sample data
x = np.random.rand(50)
y = np.random.rand(50)
data_values = np.random.randint(1, 50, 50)

# Define the sizes for the legend entries (you can adjust these as needed)
legend_sizes = [10, 20, 30, 40]

# Create the scatter plot with the main data points
plt.scatter(x, y, s=data_values, alpha=0.7)

# Create a custom legend with proxy artists (circles of different sizes) and labels
legend_elements = [plt.scatter([], [], s= size, color = 'b', label=f'Data Value: {size}') for size in legend_sizes]

# Add the legend to the plot
plt.legend(handles=legend_elements, loc='upper right')

# Add other plot elements (title, labels, etc.) if needed
plt.title('Scatter Plot with Custom Size Legend')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')

# Show the plot
plt.show()
