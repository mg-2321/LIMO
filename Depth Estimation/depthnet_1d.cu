// depthnet.cu: This is for depth estimation after the features are extracted adn 6dof pose estimation is performed on kitti dataset
#include <iostream>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#define TILE_WIDTH 16
__global__ void conv1x1Kernel(float* input, float* weights, float* output, int C_in, int H, int W) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // width
    int y = blockIdx.y * blockDim.y + threadIdx.y; // height
    if (x >= W || y >= H) return;

    int index = y * W + x;
    float val = 0.0f;
    for (int c = 0; c < C_in; c++) {
        int offset = c * H * W + y * W + x;
        val += input[offset] * weights[c];
    }
    output[index] = fmaxf(val, 0.0f); // ReLU
}

__global__ void upsampleBilinear(float* input, float* output, int H_in, int W_in, int H_out, int W_out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x; // width
    int y = blockIdx.y * blockDim.y + threadIdx.y; // height
    if (x >= W_out || y >= H_out) return;

    float scale_y = (float)H_in / H_out;
    float scale_x = (float)W_in / W_out;

    float fy = y * scale_y;
    float fx = x * scale_x;

    int y0 = (int)fy;
    int x0 = (int)fx;
    int y1 = min(y0 + 1, H_in - 1);
    int x1 = min(x0 + 1, W_in - 1);

    float dy = fy - y0;
    float dx = fx - x0;

    float top = (1 - dx) * input[y0 * W_in + x0] + dx * input[y0 * W_in + x1];
    float bottom = (1 - dx) * input[y1 * W_in + x0] + dx * input[y1 * W_in + x1];

    output[y * W_out + x] = (1 - dy) * top + dy * bottom;
}


void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (" << msg << "): " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}


int main() {
    const int H = 64, W = 64, C_in = 512;
    const int H_out = 256, W_out = 256;

    float* h_input = new float[H * W * C_in];      // Input feature map
    float* h_weights = new float[C_in];            // 1x1 convolution weights
    float* h_depthLowRes = new float[H * W];       // Output of conv1x1
    float* h_depth = new float[H_out * W_out];     // Final upsampled depth

    // Initialize input and weights (dummy)
    for (int i = 0; i < H * W * C_in; ++i) h_input[i] = static_cast<float>(rand()) / RAND_MAX;
    for (int i = 0; i < C_in; ++i) h_weights[i] = 0.001f * (rand() % 100);

    // Device memory
    float *d_input, *d_weights, *d_depthLowRes, *d_depth;
    checkCuda(cudaMalloc(&d_input, H * W * C_in * sizeof(float)), "d_input");
    checkCuda(cudaMalloc(&d_weights, C_in * sizeof(float)), "d_weights");
    checkCuda(cudaMalloc(&d_depthLowRes, H * W * sizeof(float)), "d_depthLowRes");
    checkCuda(cudaMalloc(&d_depth, H_out * W_out * sizeof(float)), "d_depth");

    // Transfer to device
    checkCuda(cudaMemcpy(d_input, h_input, H * W * C_in * sizeof(float), cudaMemcpyHostToDevice), "input");
    checkCuda(cudaMemcpy(d_weights, h_weights, C_in * sizeof(float), cudaMemcpyHostToDevice), "weights");

    // Launch conv1x1 kernel
    dim3 blockSize(TILE_WIDTH, TILE_WIDTH);
    dim3 gridSize((W + TILE_WIDTH - 1) / TILE_WIDTH, (H + TILE_WIDTH - 1) / TILE_WIDTH);
    conv1x1Kernel<<<gridSize, blockSize>>>(d_input, d_weights, d_depthLowRes, C_in, H, W);
    cudaDeviceSynchronize();

    // Launch upsampling kernel
    dim3 gridSize2((W_out + TILE_WIDTH - 1) / TILE_WIDTH, (H_out + TILE_WIDTH - 1) / TILE_WIDTH);
    upsampleBilinear<<<gridSize2, blockSize>>>(d_depthLowRes, d_depth, H, W, H_out, W_out);
    cudaDeviceSynchronize();

    // Copy back result
    checkCuda(cudaMemcpy(h_depth, d_depth, H_out * W_out * sizeof(float), cudaMemcpyDeviceToHost), "depth");

    // Save output as image (normalized)
    cv::Mat depthImage(H_out, W_out, CV_32F, h_depth);
    cv::normalize(depthImage, depthImage, 0, 255, cv::NORM_MINMAX);
    depthImage.convertTo(depthImage, CV_8U);
    cv::imwrite("depth_output.png", depthImage);

    std::cout << "Depth map saved as 'depth_output.png'\n";

    // Free
    delete[] h_input;
    delete[] h_weights;
    delete[] h_depthLowRes;
    delete[] h_depth;
    cudaFree(d_input);
    cudaFree(d_weights);
    cudaFree(d_depthLowRes);
    cudaFree(d_depth);
    return 0;
}
