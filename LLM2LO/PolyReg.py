import numpy as np
import matplotlib.pyplot as plt

# Define the inputs and outputs
x = np.array([1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
y = np.array([0.10720712, -0.01825744, 0.05092185, -0.01086908, 0.0523258, 0.02200802, 0.10961226, 0.17599457, 0.14974362, 0.36131327])

# Fit the polynomial regression function
coefficients = np.polyfit(x, y, deg=3)
polynomial = np.poly1d(coefficients)

# Get the predicted outputs
predicted_y = polynomial(x)

# Set the figure size
plt.figure(figsize=(8, 4))

# Plot the original data points
#plt.scatter(x, y, color='b', marker='o', edgecolor='black', linewidth=1, label='Original Weight')

# Plot the learned function
x_range = np.linspace(0, 1, 100)
y_range = polynomial(x_range)
plt.plot(x_range, y_range, color='r', linewidth=3)

# Set the tick size for both x and y axis
plt.xticks(fontsize=16)  # Set the x-axis tick size
plt.yticks(fontsize=16)  # Set the y-axis tick size

# Set the labels and title
plt.xlabel('Rank',fontsize=18)
plt.ylabel('Weight',fontsize=18)
#plt.title('Polynomial Regression')

# # Show the legend
# plt.legend(loc='upper right')

# Create enough room for x label
plt.subplots_adjust(bottom=0.15)

# Show grid
plt.grid(True)

# Save the plot as an image file
plt.savefig('polynomial_regression_plot.pdf', dpi=300, bbox_inches='tight')

# Show the plot
plt.show()