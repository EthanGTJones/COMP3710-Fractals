import torch
import numpy as np
import matplotlib.pyplot as plt

# PyTorch Version
print("PyTorch Version:", torch.__version__)

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# grid for computing image, subdivide the space
degree = 8
size = 3 ** degree

# Use NumPy to create a 2D array of complex numbers on [0,s**d]x[0,s**d]
X, Y = np.mgrid[0:size:1, 0:size:1]

# load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)
z = torch.zeros_like(x)

# transfer to the GPU device
x = x.to(device)
y = y.to(device)
z = z.to(device)

for i in range(degree):
    z1 = torch.logical_and(((x // (3 ** i)) % 3) == 1, ((y // (3 ** i)) % 3) == 1)
    z = torch.logical_or(z, z1)

# Plot
plt.imshow(z.cpu().numpy())
plt.tight_layout()
plt.show()
