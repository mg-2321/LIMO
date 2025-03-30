// Optimized RectifyNet CUDA Implementation

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

using namespace std;
using namespace cv;

// ------------------- Optimized CUDA Kernels -------------------

__global__ void conv1x1_kernel_opt(float* input, float* output, float* weights, int H, int W, int C_in, int C_out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H) return;

    for (int c_out = 0; c_out < C_out; ++c_out) {
        float sum = 0.0f;
        #pragma unroll
        for (int c_in = 0; c_in < C_in; ++c_in) {
            int idx = (c_in * H * W) + (y * W) + x;
            int weight_idx = (c_out * C_in) + c_in;
            sum += __ldg(&input[idx]) * __ldg(&weights[weight_idx]);
        }
        output[(c_out * H * W) + (y * W) + x] = fmaxf(0.0f, sum);
    }
}

__global__ void conv3x3_kernel_opt(float* input, float* output, float* kernel, int H, int W, int C_in, int K, int C_out) {
    extern __shared__ float shared_input[];

    int tx = threadIdx.x, ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    int pad = K / 2;

    // Load to shared memory
    for (int c = 0; c < C_in; ++c) {
        int shared_idx = c * blockDim.y * blockDim.x + ty * blockDim.x + tx;
        if (x < W && y < H) {
            shared_input[shared_idx] = input[c * H * W + y * W + x];
        }
    }
    __syncthreads();

    if (x >= W || y >= H) return;

    for (int c_out = 0; c_out < C_out; ++c_out) {
        float sum = 0.0f;
        #pragma unroll
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int i = 0; i < K; ++i) {
                for (int j = 0; j < K; ++j) {
                    int xi = tx + j - pad;
                    int yj = ty + i - pad;
                    if (xi >= 0 && xi < blockDim.x && yj >= 0 && yj < blockDim.y &&
                        x + j - pad >= 0 && x + j - pad < W && y + i - pad >= 0 && y + i - pad < H) {
                        int shared_idx = c_in * blockDim.y * blockDim.x + yj * blockDim.x + xi;
                        sum += shared_input[shared_idx] * kernel[((c_out * C_in + c_in) * K * K) + (i * K) + j];
                    }
                }
            }
        }
        output[(c_out * H * W) + (y * W) + x] = fmaxf(0.0f, sum);
    }
}

__global__ void conv1x1_final_kernel_opt(float* input, float* output, float* weights, int H, int W, int C_in, int C_out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= W || y >= H) return;

    for (int c_out = 0; c_out < C_out; ++c_out) {
        float sum = 0.0f;
        #pragma unroll
        for (int c_in = 0; c_in < C_in; ++c_in) {
            int idx = (c_in * H * W) + (y * W) + x;
            int weight_idx = (c_out * C_in) + c_in;
            sum += __ldg(&input[idx]) * __ldg(&weights[weight_idx]);
        }
        output[(c_out * H * W) + (y * W) + x] = sum;
    }
}

__global__ void global_avg_pool_kernel_opt(float* input, float* output, int H, int W, int C_in) {
    int c = threadIdx.x;
    if (c >= C_in) return;

    float sum = 0.0f;
    for (int y = 0; y < H; ++y) {
        #pragma unroll
        for (int x = 0; x < W; ++x) {
            sum += input[(c * H * W) + (y * W) + x];
        }
    }
    output[c] = sum / (H * W);
}

// ------------------- Rest of Implementation -------------------

// You can request me to proceed now to provide the rest of the complete working code!
