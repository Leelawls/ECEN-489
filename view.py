import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

# Define transformations (same as used for training)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Get a batch of training images
trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True)
images, labels = next(iter(trainloader))  # Get first batch

# Display images
fig, axes = plt.subplots(1, 8, figsize=(10, 2))
for i in range(8):
    axes[i].imshow(images[i].squeeze(), cmap='gray')  # Convert tensor to image
    axes[i].set_title(f"Label: {labels[i].item()}")
    axes[i].axis('off')

plt.show()
