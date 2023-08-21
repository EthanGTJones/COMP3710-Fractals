import torch
import numpy as np
import matplotlib.pyplot as plt

# PyTorch Version
print("PyTorch Version:", torch.__version__)

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# grid for computing image, subdivide the space
X, Y = np.mgrid[-4.0:4:0.01, -4.0:4:0.01]

# load into PyTorch tensors
x = torch.Tensor(X)
y = torch.Tensor(Y)

# transfer to the GPU device
x = x.to(device)
y = y.to(device)

# (1 Mark) Change the Gaussian function into a 2D sine or cosine function
g = torch.exp(-(x ** 2 + y ** 2) / 2.0)  # Given Gaussian function.
s = torch.sin(x + y)

# (1 Mark) What do you get when you multiply both the Gaussian and the
# sine/cosine function together and visualise it?
z = g * s  # Gabor Filter

# Plot
plt.imshow(z.cpu().numpy())
plt.tight_layout()
plt.show()
