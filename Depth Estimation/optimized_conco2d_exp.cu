// conv2d.cu
#include <iostream>
#include <cuda_runtime.h>

#define TILE_WIDTH 16
#define KERNEL_SIZE 3
#define CHANNELS_IN 64
#define CHANNELS_OUT 64

__constant__ float d_kernel[CHANNELS_OUT * CHANNELS_IN * KERNEL_SIZE * KERNEL_SIZE];

__global__ void conv2dTiledShared(float* input, float* output,
                                  int H, int W,
                                  int C_in, int C_out) {
    __shared__ float tile[TILE_WIDTH + KERNEL_SIZE - 1][TILE_WIDTH + KERNEL_SIZE - 1];

    int tx = threadIdx.x, ty = threadIdx.y;
    int x = blockIdx.x * TILE_WIDTH + tx;
    int y = blockIdx.y * TILE_WIDTH + ty;

    for (int co = 0; co < C_out; co++) {
        float sum = 0.0f;
        for (int ci = 0; ci < C_in; ci++) {
            for (int i = 0; i < KERNEL_SIZE; i++) {
                for (int j = 0; j < KERNEL_SIZE; j++) {
                    int in_x = x + j - KERNEL_SIZE / 2;
                    int in_y = y + i - KERNEL_SIZE / 2;

                    float val = 0.0f;
                    if (in_x >= 0 && in_x < W && in_y >= 0 && in_y < H) {
                        int input_idx = ci * H * W + in_y * W + in_x;
                        val = input[input_idx];
                    }

                    int weight_idx = co * C_in * KERNEL_SIZE * KERNEL_SIZE +
                                     ci * KERNEL_SIZE * KERNEL_SIZE +
                                     i * KERNEL_SIZE + j;

                    sum += val * d_kernel[weight_idx];
                }
            }
        }

        sum = fmaxf(sum, 0.0f); // ReLU
        if (x < W && y < H) {
            int output_idx = co * H * W + y * W + x;
            output[output_idx] = sum;
        }
    }
}

// ------------------------ HOST CODE TO DEMO ------------------------

void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (" << msg << "): " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int H = 64, W = 64;
    const int C_in = CHANNELS_IN;
    const int C_out = CHANNELS_OUT;

    size_t input_size = C_in * H * W * sizeof(float);
    size_t output_size = C_out * H * W * sizeof(float);
    size_t kernel_size = C_out * C_in * KERNEL_SIZE * KERNEL_SIZE * sizeof(float);

    float* h_input = new float[C_in * H * W];
    float* h_output = new float[C_out * H * W];
    float* h_kernel = new float[C_out * C_in * KERNEL_SIZE * KERNEL_SIZE];

    // Initialize inputs and weights
    for (int i = 0; i < C_in * H * W; ++i) h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < C_out * C_in * KERNEL_SIZE * KERNEL_SIZE; ++i)
        h_kernel[i] = static_cast<float>(rand()) / RAND_MAX * 0.1f;

    float *d_input, *d_output;
    checkCuda(cudaMalloc(&d_input, input_size), "d_input");
    checkCuda(cudaMalloc(&d_output, output_size), "d_output");

    // Copy to device
    checkCuda(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice), "input");
    checkCuda(cudaMemcpyToSymbol(d_kernel, h_kernel, kernel_size), "kernel");

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim((W + TILE_WIDTH - 1) / TILE_WIDTH, (H + TILE_WIDTH - 1) / TILE_WIDTH);

    conv2dTiledShared<<<gridDim, blockDim>>>(d_input, d_output, H, W, C_in, C_out);
    cudaDeviceSynchronize();

    checkCuda(cudaMemcpy(h_output, d_output, output_size, cudaMemcpyDeviceToHost), "output");

    std::cout << "2D Convolution complete. Sample output[0]: " << h_output[0] << std::endl;

    // Clean up
    delete[] h_input;
    delete[] h_output;
    delete[] h_kernel;
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
}
