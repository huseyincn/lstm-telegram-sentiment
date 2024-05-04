import numpy as np

def get_largest_indices(arr, n):
    """Return the indices of the n largest elements in arr."""
    return np.argsort(arr.flatten())[-n:][::-1]

# Your array
arr = np.array([[9.9994838e-01, 5.6974109e-06, 6.0020470e-06, 1.4201040e-05, 2.1777305e-05, 3.9231932e-06]])

# Get the indices of the 2 largest elements
indices = get_largest_indices(arr, 2)

print(indices)