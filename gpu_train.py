import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import conv2d_cuda

import time

torch.manual_seed(21)
# Define transformation (convert image to tensor & normalize)
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: 1 - x),  # Invert colors
    transforms.Normalize((0.5,), (0.5,))
])


# Load MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)  # Increased batch size

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1_kernel = nn.Parameter(torch.randn(32, 1, 3, 3, device="cuda"))
        self.conv2_kernel = nn.Parameter(torch.randn(64, 32, 3, 3, device="cuda"))
        self.conv3_kernel = nn.Parameter(torch.randn(128, 64, 3, 3, device="cuda"))

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.3)

        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x_out = torch.zeros_like(x, device="cuda")  # Placeholder for CUDA computation

        x = self.pool(torch.relu(self.bn1(conv2d_cuda.conv2d_cuda(x, self.conv1_kernel, x_out, 1, 32, 28, 28, 3, 1, 1))))
        x = self.pool(torch.relu(self.bn2(conv2d_cuda.conv2d_cuda(x, self.conv2_kernel, x_out, 32, 64, 14, 14, 3, 1, 1))))
        x = self.pool(torch.relu(self.bn3(conv2d_cuda.conv2d_cuda(x, self.conv3_kernel, x_out, 64, 128, 7, 7, 3, 1, 1))))

        x = x.view(-1, 128 * 3 * 3)  # Flatten
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x



if __name__ == "__main__":
    # Create model, loss function, and optimizer
    start_time = time.perf_counter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)  # Added weight decay for better optimization

    # Train model
    for epoch in range(10):  # Increased epochs for better training
        model.train()
        running_loss = 0.0

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/10], Loss: {running_loss/len(trainloader):.4f}")

    # Evaluate model
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

    #Calculate total time
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time} seconds")
    
    # Save the model state dictionary
    torch.save(model.state_dict(), "gpu_cnn_model.pth")
