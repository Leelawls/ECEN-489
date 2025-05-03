// MPI + CUDA convolution over multiple processes (Grace-compatible)
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <cuda_runtime.h>
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"
#define KERNEL_SIZE 3
#define BLOCK_SIZE 16
#define NUM_IMAGES 8192  // Increase workload for better GPU utilization

__constant__ float d_kernel[KERNEL_SIZE * KERNEL_SIZE];

__global__ void conv2d_shared(float* input, float* output, int width, int height) {
    __shared__ float tile[BLOCK_SIZE + KERNEL_SIZE - 1][BLOCK_SIZE + KERNEL_SIZE - 1];
    int tx = threadIdx.x, ty = threadIdx.y;
    int x = blockIdx.x * BLOCK_SIZE + tx;
    int y = blockIdx.y * BLOCK_SIZE + ty;
    int halo = KERNEL_SIZE / 2;

    if (x < width && y < height)
        tile[ty][tx] = input[y * width + x];
    else
        tile[ty][tx] = 0.0f;

    __syncthreads();

    float sum = 0.0f;
    if (tx >= halo && ty >= halo && tx < BLOCK_SIZE + halo && ty < BLOCK_SIZE + halo &&
        x >= halo && x < (width - halo) && y >= halo && y < (height - halo)) {
        for (int i = 0; i < KERNEL_SIZE; ++i)
            for (int j = 0; j < KERNEL_SIZE; ++j)
                sum += tile[ty - halo + i][tx - halo + j] * d_kernel[i * KERNEL_SIZE + j];
        output[y * width + x] = sum;
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    cudaSetDevice(rank % deviceCount);

    int width, height, channels;
    unsigned char* img = NULL;
    if (rank == 0) {
        img = stbi_load("face.jpg", &width, &height, &channels, 1);
        if (!img) {
            printf("Failed to load image.\n");
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    }

    MPI_Bcast(&width, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&height, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int img_size = width * height;
    float* h_input = (float*)malloc(img_size * sizeof(float));

    if (rank == 0) {
        for (int i = 0; i < img_size; i++)
            h_input[i] = img[i] / 255.0f;
    }

    MPI_Bcast(h_input, img_size, MPI_FLOAT, 0, MPI_COMM_WORLD);

    float h_kernel[KERNEL_SIZE * KERNEL_SIZE] = {
        1.0f/9, 1.0f/9, 1.0f/9,
        1.0f/9, 1.0f/9, 1.0f/9,
        1.0f/9, 1.0f/9, 1.0f/9
    };
    cudaMemcpyToSymbol(d_kernel, h_kernel, sizeof(float) * KERNEL_SIZE * KERNEL_SIZE);

    float *d_input, *d_output;
    cudaMalloc(&d_input, img_size * sizeof(float));
    cudaMalloc(&d_output, img_size * sizeof(float));
    cudaMemcpy(d_input, h_input, img_size * sizeof(float), cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE + KERNEL_SIZE - 1, BLOCK_SIZE + KERNEL_SIZE - 1);
    dim3 dimGrid((width + BLOCK_SIZE - 1) / BLOCK_SIZE, (height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    int local_images = NUM_IMAGES / size;
    for (int i = 0; i < local_images; i++)
        conv2d_shared<<<dimGrid, dimBlock>>>(d_input, d_output, width, height);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float local_time = 0;
    cudaEventElapsedTime(&local_time, start, stop);

    float max_time = 0;
    MPI_Reduce(&local_time, &max_time, 1, MPI_FLOAT, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("[MPI+CUDA] Processed %d images with %d ranks.\n", NUM_IMAGES, size);
        printf("Max time across ranks: %.2f ms\n", max_time);
        printf("Avg per image: %.5f ms\n", max_time / NUM_IMAGES);
    }

    free(h_input);
    if (img) stbi_image_free(img);
    cudaFree(d_input);
    cudaFree(d_output);

    MPI_Finalize();
    return 0;
}

