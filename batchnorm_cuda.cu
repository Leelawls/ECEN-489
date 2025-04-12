#include <iostream>
#include <cuda_runtime.h>

__global__ void batchnorm_inference_kernel(
    const float* input, float* output,
    const float* mean, const float* var,
    const float* gamma, const float* beta,
    int N, int C, int H, int W, float epsilon
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = N * C * H * W;
    if (idx >= total_elements) return;

    int w = idx % W;
    int h = (idx / W) % H;
    int c = (idx / (H * W)) % C;
    int n = idx / (C * H * W);

    int offset = ((n * C + c) * H + h) * W + w;

    float x = input[offset];
    float norm = (x - mean[c]) / sqrtf(var[c] + epsilon);
    output[offset] = gamma[c] * norm + beta[c];
}

void run_batchnorm(
    const float* h_input, float* h_output,
    const float* h_mean, const float* h_var,
    const float* h_gamma, const float* h_beta,
    int N, int C, int H, int W, float epsilon
) {
    int numel = N * C * H * W;
    size_t size = numel * sizeof(float);
    size_t csize = C * sizeof(float);

    float *d_input, *d_output, *d_mean, *d_var, *d_gamma, *d_beta;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_mean, csize);
    cudaMalloc(&d_var, csize);
    cudaMalloc(&d_gamma, csize);
    cudaMalloc(&d_beta, csize);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mean, h_mean, csize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_var, h_var, csize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, csize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, csize, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (numel + threads - 1) / threads;
    // Timing setup
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch kernel
    batchnorm_inference_kernel<<<blocks, threads>>>(
        d_input, d_output, d_mean, d_var, d_gamma, d_beta, N, C, H, W, epsilon
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "\nBatchNorm execution time: " << milliseconds << " ms\n";


    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    // Cleanup
    cudaFree(d_input); cudaFree(d_output);
    cudaFree(d_mean); cudaFree(d_var);
    cudaFree(d_gamma); cudaFree(d_beta);
}

int main() {
    const int N = 1, C = 3, H = 4, W = 4;
    float epsilon = 1e-5f;
    int numel = N * C * H * W;

    float h_input[numel];
    float h_output[numel];

    float h_mean[C] = {0.5, 0.4, 0.3};
    float h_var[C] = {0.1, 0.2, 0.3};
    float h_gamma[C] = {1.0, 1.0, 1.0};
    float h_beta[C] = {0.0, 0.0, 0.0};

    // Fill input with dummy values
    for (int i = 0; i < numel; ++i) {
        h_input[i] = (i % 10) / 10.0f;
    }

    run_batchnorm(h_input, h_output, h_mean, h_var, h_gamma, h_beta, N, C, H, W, epsilon);

    std::cout << "BatchNorm output:\n";
    for (int i = 0; i < numel; ++i) {
        std::cout << h_output[i] << " ";
        if ((i + 1) % W == 0) std::cout << "\n";
        if ((i + 1) % (H * W) == 0) std::cout << "--------\n";
    }

    return 0;
}
