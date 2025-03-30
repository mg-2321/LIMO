//To compile:nvcc -o resnet_encoder /home/NETID/gayat23/cuda_assignment/sc_depth_pl/resnet_encoder.cu `pkg-config --cflags --libs opencv` -std=c++17 -lstdc++fs
//To compile:./resnet_encoder /home/NETID/gayat23/cuda_/sc_depth_pl/datasets/__pycache__/kitti/training/2011_09_26_drive_0001_sync_02/ /home/NETID/gayat23/cuda_/sc_depth_pl/datasets/__pycache__/kitti/training/2011_09_26_drive_0001_sync_02/frame_index.txt

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <string>

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

// Optimized CUDA kernel with shared memory & coalesced access
__global__ void conv2d_shared_kernel(float* input, float* output, float* kernel, int H, int W, int C_in, int K, int C_out) {
    extern __shared__ float shared_input[];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    int pad = K / 2;

    // Load input to shared memory (coalesced)
    for (int c = 0; c < C_in; ++c) {
        int shared_idx = c * blockDim.y * blockDim.x + ty * blockDim.x + tx;
        if (x < W && y < H) {
            shared_input[shared_idx] = input[c * H * W + y * W + x];
        }
    }
    __syncthreads();

    if (x >= W || y >= H) return;

    // Convolution computation
    for (int c_out = 0; c_out < C_out; ++c_out) {
        float sum = 0.0f;
        for (int c = 0; c < C_in; ++c) {
            for (int i = 0; i < K; ++i) {
                for (int j = 0; j < K; ++j) {
                    int xi = tx + j - pad;
                    int yj = ty + i - pad;
                    if (xi >= 0 && xi < blockDim.x && yj >= 0 && yj < blockDim.y &&
                        x + j - pad >= 0 && x + j - pad < W && y + i - pad >= 0 && y + i - pad < H) {
                        int shared_idx = c * blockDim.y * blockDim.x + yj * blockDim.x + xi;
                        sum += shared_input[shared_idx] * kernel[((c_out * C_in + c) * K * K) + (i * K) + j];
                    }
                }
            }
        }
        output[(c_out * H * W) + (y * W) + x] = fmaxf(0.0f, sum); // ReLU
    }
}

vector<int> load_frame_indices(const string& txt_path) {
    vector<int> indices;
    ifstream infile(txt_path);
    string line;
    while (getline(infile, line)) {
        indices.push_back(stoi(line));
    }
    return indices;
}

void save_feature_to_csv(const vector<float>& features, int C_out, int H, int W, int frame_idx, const string& output_folder) {
    string csv_name = output_folder + "/feature_" + to_string(frame_idx) + ".csv";
    ofstream csv(csv_name);
    for (int c = 0; c < C_out; ++c) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                csv << features[(c * H * W) + (y * W) + x];
                if (x != W - 1) csv << ",";
            }
            csv << endl;
        }
    }
    csv.close();
}

int main(int argc, char** argv) {
    if (argc < 3) {
        cout << "Usage: " << argv[0] << " <image_folder> <frame_index.txt>" << endl;
        return -1;
    }

    string image_folder = argv[1];
    string frame_index_path = argv[2];
    vector<int> valid_indices = load_frame_indices(frame_index_path);

    cout << "Found " << valid_indices.size() << " valid frames." << endl;

    int kernel_size = 3;
    int C_in = 3;
    int C_out = 16;
    float* h_kernel = new float[C_out * C_in * kernel_size * kernel_size];

    for (int i = 0; i < C_out * C_in * kernel_size * kernel_size; ++i) {
        h_kernel[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    float* d_kernel;
    cudaMalloc(&d_kernel, C_out * C_in * kernel_size * kernel_size * sizeof(float));
    cudaMemcpy(d_kernel, h_kernel, C_out * C_in * kernel_size * kernel_size * sizeof(float), cudaMemcpyHostToDevice);

    fs::path output_folder = "features_output";
    if (!fs::exists(output_folder)) fs::create_directory(output_folder);

    ofstream timing_log("timing_log.csv");
    timing_log << "Frame,KernelTime(ms),MemTransferTime(ms)\n";

    for (int idx : valid_indices) {
        char filename[256];
        sprintf(filename, "%s/%010d.jpg", image_folder.c_str(), idx);
        Mat img = imread(filename);
        if (img.empty()) continue;
        resize(img, img, Size(224, 224));
        img.convertTo(img, CV_32FC3, 1.0 / 255);

        int H = img.rows, W = img.cols;
        size_t img_size = C_in * H * W * sizeof(float);

        // Pinned memory
        float* h_input;
        cudaMallocHost(&h_input, img_size);
        memcpy(h_input, img.data, img_size);

        float* d_input;
        float* d_output;
        cudaMalloc(&d_input, img_size);
        cudaMalloc(&d_output, C_out * H * W * sizeof(float));

        // Timing memory transfer
        cudaEvent_t start_mem, stop_mem;
        cudaEventCreate(&start_mem);
        cudaEventCreate(&stop_mem);
        cudaEventRecord(start_mem);

        cudaMemcpy(d_input, h_input, img_size, cudaMemcpyHostToDevice);

        cudaEventRecord(stop_mem);
        cudaEventSynchronize(stop_mem);
        float mem_ms = 0;
        cudaEventElapsedTime(&mem_ms, start_mem, stop_mem);

        // Kernel timing
        cudaEvent_t start_kernel, stop_kernel;
        cudaEventCreate(&start_kernel);
        cudaEventCreate(&stop_kernel);

        dim3 block(16, 16);
        dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
        size_t shared_mem_size = C_in * block.x * block.y * sizeof(float);

        cudaEventRecord(start_kernel);
        conv2d_shared_kernel<<<grid, block, shared_mem_size>>>(d_input, d_output, d_kernel, H, W, C_in, kernel_size, C_out);
        cudaEventRecord(stop_kernel);
        cudaEventSynchronize(stop_kernel);

        float kernel_ms = 0;
        cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel);

        vector<float> output(C_out * H * W);
        cudaMemcpy(output.data(), d_output, C_out * H * W * sizeof(float), cudaMemcpyDeviceToHost);
        
        cout << "Frame: " << idx << " KernelTime: " << kernel_ms << " ms MemTransferTime: " << mem_ms << " ms" << endl;
        timing_log << idx << "," << kernel_ms << "," << mem_ms << "\n";

        save_feature_to_csv(output, C_out, H, W, idx, output_folder.string());

        cudaFree(d_input);
        cudaFree(d_output);
        cudaFreeHost(h_input);
    }

    cudaFree(d_kernel);
    delete[] h_kernel;
    timing_log.close();
    cout << "Feature extraction completed. Timing logged to timing_log.csv" << endl;

    return 0;
}
