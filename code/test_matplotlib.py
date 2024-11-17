# Importing necessary libraries
import matplotlib.pyplot as plt
import numpy as np

# Generating data
x = np.linspace(0, 10, 100)  # 100 evenly spaced values from 0 to 10
y = np.sin(x)                # Sine of each value in x

# Creating the plot
plt.figure(figsize=(8, 4))    # Set the figure size
plt.plot(x, y, label='sin(x)', color='blue')  # Plot y vs. x with label and color

# Adding titles and labels
plt.title("Simple Sine Wave Plot")       # Title of the plot
plt.xlabel("x values")                   # Label for x-axis
plt.ylabel("sin(x)")                     # Label for y-axis

# Adding a legend
plt.legend()                             # Display legend with 'sin(x)' label

# Displaying the plot
plt.show()
