import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import time

# Load image in grayscale
img = cv2.imread("face.jpg", cv2.IMREAD_GRAYSCALE)

img = img / 255.0

kernel = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
])

start = time.perf_counter()
convolved = convolve2d(img, kernel, mode='same', boundary='symm')
end = time.perf_counter()
print(f"Convolution time: {(end - start) * 1000:.2f} ms")

#Highs and lows are exagerated so it appears better when looking but convolutions would stop here - for demo only
p_low, p_high = np.percentile(convolved, (1, 99))  
convolved = np.clip(convolved, p_low, p_high)

convolved = (convolved - p_low) / (p_high - p_low)
convolved_uint8 = (convolved * 255).astype(np.uint8)
cv2.imwrite("convolved_output.png", convolved_uint8)

