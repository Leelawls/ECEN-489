#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define INPUT_SIZE 784
#define HIDDEN_SIZE 64
#define OUTPUT_SIZE 10

// Shared-memory FC kernel
__global__ void fc_layer_shared(
    const float* input, const float* weight, const float* bias,
    float* output, int in_size, int out_size, bool use_relu
) {
    extern __shared__ float s_input[];
    int tid = threadIdx.x;
    int neuron_id = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input to shared memory once per block
    for (int i = tid; i < in_size; i += blockDim.x) {
        s_input[i] = input[i];
    }
    __syncthreads();

    // Compute output neuron
    if (neuron_id < out_size) {
        float sum = 0.0f;
        for (int j = 0; j < in_size; j++) {
            sum += s_input[j] * weight[neuron_id * in_size + j];
        }
        sum += bias[neuron_id];
        if (use_relu)
            sum = fmaxf(0.0f, sum);
        output[neuron_id] = sum;
    }
}

int main() {
    // -------------------------------
    // Host allocations and init
    // -------------------------------
    float h_input[INPUT_SIZE];
    float h_fc1_weight[HIDDEN_SIZE * INPUT_SIZE];
    float h_fc1_bias[HIDDEN_SIZE];
    float h_fc2_weight[OUTPUT_SIZE * HIDDEN_SIZE];
    float h_fc2_bias[OUTPUT_SIZE];
    float h_output[OUTPUT_SIZE];

    for (int i = 0; i < INPUT_SIZE; i++)
        h_input[i] = float(i + 1);

    for (int i = 0; i < HIDDEN_SIZE * INPUT_SIZE; i++)
        h_fc1_weight[i] = 0.01f;
    for (int i = 0; i < HIDDEN_SIZE; i++)
        h_fc1_bias[i] = 0.0f;

    for (int i = 0; i < OUTPUT_SIZE * HIDDEN_SIZE; i++)
        h_fc2_weight[i] = 0.01f;
    for (int i = 0; i < OUTPUT_SIZE; i++)
        h_fc2_bias[i] = 0.0f;

    // -------------------------------
    // Device allocations
    // -------------------------------
    float *d_input, *d_fc1_weight, *d_fc1_bias, *d_hidden;
    float *d_fc2_weight, *d_fc2_bias, *d_output;

    cudaMalloc(&d_input, INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_fc1_weight, HIDDEN_SIZE * INPUT_SIZE * sizeof(float));
    cudaMalloc(&d_fc1_bias, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_hidden, HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_fc2_weight, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float));
    cudaMalloc(&d_fc2_bias, OUTPUT_SIZE * sizeof(float));
    cudaMalloc(&d_output, OUTPUT_SIZE * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_input, h_input, INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc1_weight, h_fc1_weight, HIDDEN_SIZE * INPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc1_bias, h_fc1_bias, HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2_weight, h_fc2_weight, OUTPUT_SIZE * HIDDEN_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_fc2_bias, h_fc2_bias, OUTPUT_SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // -------------------------------
    // Timing
    // -------------------------------
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // FC1 + ReLU
    fc_layer_shared<<<(HIDDEN_SIZE + 255) / 256, 256, INPUT_SIZE * sizeof(float)>>>(
        d_input, d_fc1_weight, d_fc1_bias, d_hidden,
        INPUT_SIZE, HIDDEN_SIZE, true
    );

    // FC2 (no ReLU)
    fc_layer_shared<<<(OUTPUT_SIZE + 255) / 256, 256, HIDDEN_SIZE * sizeof(float)>>>(
        d_hidden, d_fc2_weight, d_fc2_bias, d_output,
        HIDDEN_SIZE, OUTPUT_SIZE, false
    );

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsed_ms = 0;
    cudaEventElapsedTime(&elapsed_ms, start, stop);

    // -------------------------------
    // Copy and print result
    // -------------------------------
    cudaMemcpy(h_output, d_output, OUTPUT_SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    printf("Output logits (shared memory):\n");
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        printf("%.4f ", h_output[i]);
    }
    printf("\nForward pass time: %.3f ms\n", elapsed_ms);

    // -------------------------------
    // Cleanup
    // -------------------------------
    cudaFree(d_input);
    cudaFree(d_fc1_weight);
    cudaFree(d_fc1_bias);
    cudaFree(d_hidden);
    cudaFree(d_fc2_weight);
    cudaFree(d_fc2_bias);
    cudaFree(d_output);

    return 0;
}

