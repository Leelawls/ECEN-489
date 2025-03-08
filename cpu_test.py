import torch
from PIL import Image
import torchvision.transforms as transforms
from cpu_train import CNN  # Import CNN model

# Load model
model = CNN()
model.load_state_dict(torch.load("cpu_cnn_model.pth"))
model.eval()  # Set model to evaluation mode

# Define transformation for a single image
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure grayscale
    transforms.Resize((28, 28)),  # Resize to 28x28
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize
])

# Load and preprocess a single image
image = Image.open("digit.png").convert("L")  # Convert to grayscale
image = transform(image)  # Apply transformations
image = image.unsqueeze(0)  # Add batch dimension (1, 1, 28, 28)

# Ensure proper tensor shape before passing to model
print(f"Image shape after processing: {image.shape}")  # Should be [1, 1, 28, 28]

# Make a prediction
with torch.no_grad():
    output = model(image)  # Forward pass
    predicted_label = torch.argmax(output, dim=1)  # Get the class with highest probability

# Print predicted digit
print(f"Predicted digit: {predicted_label.item()}")
