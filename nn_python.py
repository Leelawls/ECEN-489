import torch
import torch.nn as nn
import time

#input should be flattened 28x28 pixels so 784 total length
input_tensor = torch.arange(1, 785, dtype=torch.float32).view(1, 784)


class ClassifierOnly(nn.Module):
    def __init__(self):
        super(ClassifierOnly, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 10)

        # Manually initialize weights and biases to match CUDA
        with torch.no_grad():
            self.fc1.weight.fill_(0.01)
            self.fc1.bias.zero_()
            self.fc2.weight.fill_(0.01)
            self.fc2.bias.zero_()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = ClassifierOnly()

start = time.perf_counter()
output = model(input_tensor)
end = time.perf_counter()

print("Output logits from PyTorch:")
print(output.detach().numpy().flatten())

print(f"Forward pass time: {(end - start) * 1000:.3f} ms")
