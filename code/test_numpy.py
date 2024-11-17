import numpy as np

# Creating arrays from lists
array1 = np.array([1, 2, 3, 4, 5])    # 1D array
array2 = np.array([[1, 2, 3], [4, 5, 6]])  # 2D array

# Creating arrays with specific values
zeros = np.zeros((2, 3))        # 2x3 array of zeros
ones = np.ones((3, 2))          # 3x2 array of ones
identity_matrix = np.eye(3)     # 3x3 identity matrix
random_values = np.random.rand(3, 3)  # 3x3 array with random values

print("1D Array:", array1)
print("2D Array:\n", array2)
print("Zeros:\n", zeros)
print("Ones:\n", ones)
print("Identity Matrix:\n", identity_matrix)
print("Random Values:\n", random_values)
