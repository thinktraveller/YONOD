import numpy as np

# Load the .npz file
data = np.load("Suzuki_MFP.npz")

# Check what arrays are stored
print("Keys in file:", data.files)

# Extract the 'X' array
X = data['X']
y=data['y']
# Print the shape of X
print("Shape of X:", X.shape)

# Print the first 100 elements of the first row
print("First 200 elements of X[30]:", X[0][:200])
print("First 200 elements of X[30]:", y[0])


