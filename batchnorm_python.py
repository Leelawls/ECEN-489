import torch
import torch.nn as nn
import numpy as np
import time

N, C, H, W = 1, 3, 4, 4
epsilon = 1e-5

# Match input values from .cu (e.g., i % 10 / 10.0f)
input_data = np.array([(i % 10) / 10.0 for i in range(N * C * H * W)], dtype=np.float32)
input_tensor = torch.tensor(input_data.reshape(N, C, H, W))

bn = nn.BatchNorm2d(C)

# Set fixed parameters (same as .cu)
with torch.no_grad():
    bn.running_mean = torch.tensor([0.5, 0.4, 0.3])
    bn.running_var = torch.tensor([0.1, 0.2, 0.3])
    bn.weight = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))
    bn.bias = nn.Parameter(torch.tensor([0.0, 0.0, 0.0]))

bn.eval()

start = time.perf_counter()
out = bn(input_tensor)
end = time.perf_counter()
print(f"\nBatchNorm execution time: {(end - start) * 1000:.2f} ms\n")

out_np = out.squeeze(0).detach().numpy() 

for c in range(C):
    print(f"Channel {c}:")
    for h in range(H):
        for w in range(W):
            print(f"{out_np[c, h, w]:.4f}", end=' ')
        print()
    print("--------")
