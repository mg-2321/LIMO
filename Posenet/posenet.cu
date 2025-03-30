#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cuda_runtime.h>

using namespace std;

// Parameters
const int H = 224;
const int W = 224;
const int C_in = 16;
const int reduced_C = 256;

// ---------- CUDA Kernels ----------

// 1x1 Conv
__global__ void conv1x1_kernel(float* input, float* output, float* weights, int H, int W, int C_in, int C_out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    for (int c_out = 0; c_out < C_out; ++c_out) {
        float sum = 0.0f;
        for (int c_in = 0; c_in < C_in; ++c_in) {
            int idx = (c_in * H * W) + (y * W) + x;
            int weight_idx = (c_out * C_in) + c_in;
            sum += input[idx] * weights[weight_idx];
        }
        output[(c_out * H * W) + (y * W) + x] = fmaxf(0.0f, sum); // ReLU
    }
}

// 3x3 Conv
__global__ void conv3x3_kernel(float* input, float* output, float* kernel, int H, int W, int C_in, int K, int C_out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int pad = K / 2;
    if (x >= W || y >= H) return;

    for (int c_out = 0; c_out < C_out; ++c_out) {
        float sum = 0.0f;
        for (int c_in = 0; c_in < C_in; ++c_in) {
            for (int i = 0; i < K; ++i) {
                for (int j = 0; j < K; ++j) {
                    int xi = x + i - pad;
                    int yj = y + j - pad;
                    if (xi >= 0 && xi < W && yj >= 0 && yj < H) {
                        int idx = (c_in * H * W) + (yj * W) + xi;
                        sum += input[idx] * kernel[((c_out * C_in + c_in) * K * K) + (i * K) + j];
                    }
                }
            }
        }
        output[(c_out * H * W) + (y * W) + x] = fmaxf(0.0f, sum); // ReLU
    }
}

// Final 1x1 Conv (Output 6 channels)
__global__ void conv1x1_final_kernel(float* input, float* output, float* weights, int H, int W, int C_in, int C_out) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    for (int c_out = 0; c_out < C_out; ++c_out) {
        float sum = 0.0f;
        for (int c_in = 0; c_in < C_in; ++c_in) {
            int idx = (c_in * H * W) + (y * W) + x;
            int weight_idx = (c_out * C_in) + c_in;
            sum += input[idx] * weights[weight_idx];
        }
        output[(c_out * H * W) + (y * W) + x] = sum;
    }
}

// Global Avg Pool
__global__ void global_avg_pool_kernel(float* input, float* output, int H, int W, int C_in) {
    int c = threadIdx.x;
    if (c >= C_in) return;
    float sum = 0.0f;
    for (int y = 0; y < H; ++y)
        for (int x = 0; x < W; ++x)
            sum += input[(c * H * W) + (y * W) + x];
    output[c] = sum / (H * W);
}

// ---------- Helper Functions ----------

void load_feature_from_csv(const string& csv_path, vector<float>& feature) {
    ifstream file(csv_path);
    string line;
    int idx = 0;
    while (getline(file, line)) {
        size_t pos = 0;
        while ((pos = line.find(',')) != string::npos) {
            feature[idx++] = stof(line.substr(0, pos));
            line.erase(0, pos + 1);
        }
        feature[idx++] = stof(line); // last val
    }
}

void save_pose_to_csv(const vector<float>& pose, const string& output_path) {
    ofstream out_csv(output_path);
    for (int i = 0; i < pose.size(); ++i) {
        out_csv << pose[i];
        if (i != pose.size() - 1) out_csv << ",";
    }
    out_csv << endl;
    out_csv.close();
}

// ---------- Main Function ----------

int main(int argc, char** argv) {
    if (argc < 4) {
        cout << "Usage: " << argv[0] << " <feature_csv_img1> <feature_csv_img2> <output_pose_csv>" << endl;
        return -1;
    }

    string feature_csv1 = argv[1];
    string feature_csv2 = argv[2];
    string output_pose = argv[3];

    // Step 1: Load Features
    vector<float> feature1(C_in * H * W);
    vector<float> feature2(C_in * H * W);
    load_feature_from_csv(feature_csv1, feature1);
    load_feature_from_csv(feature_csv2, feature2);

    // Step 2: Allocate GPU memory
    float *d_feat1, *d_feat2, *d_feat1_reduced, *d_feat2_reduced;
    cudaMalloc(&d_feat1, C_in * H * W * sizeof(float));
    cudaMalloc(&d_feat2, C_in * H * W * sizeof(float));
    cudaMalloc(&d_feat1_reduced, reduced_C * H * W * sizeof(float));
    cudaMalloc(&d_feat2_reduced, reduced_C * H * W * sizeof(float));

    cudaMemcpy(d_feat1, feature1.data(), C_in * H * W * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_feat2, feature2.data(), C_in * H * W * sizeof(float), cudaMemcpyHostToDevice);

    // Step 3: 1x1 Convs
    vector<float> conv1x1_weights(reduced_C * C_in);
    for (auto& w : conv1x1_weights) w = static_cast<float>(rand()) / RAND_MAX;
    float* d_conv1x1_weights;
    cudaMalloc(&d_conv1x1_weights, reduced_C * C_in * sizeof(float));
    cudaMemcpy(d_conv1x1_weights, conv1x1_weights.data(), reduced_C * C_in * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    conv1x1_kernel<<<grid, block>>>(d_feat1, d_feat1_reduced, d_conv1x1_weights, H, W, C_in, reduced_C);
    conv1x1_kernel<<<grid, block>>>(d_feat2, d_feat2_reduced, d_conv1x1_weights, H, W, C_in, reduced_C);
    cudaDeviceSynchronize();

    // Step 4: Concatenate
    float* d_concat;
    cudaMalloc(&d_concat, 2 * reduced_C * H * W * sizeof(float));
    cudaMemcpy(d_concat, d_feat1_reduced, reduced_C * H * W * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_concat + reduced_C * H * W, d_feat2_reduced, reduced_C * H * W * sizeof(float), cudaMemcpyDeviceToDevice);

    // Step 5: 3x3 Conv
    const int conv_out_C = 256;
    float* d_conv1_out;
    cudaMalloc(&d_conv1_out, conv_out_C * H * W * sizeof(float));
    vector<float> conv3x3_weights(conv_out_C * 2 * reduced_C * 3 * 3);
    for (auto& w : conv3x3_weights) w = static_cast<float>(rand()) / RAND_MAX;
    float* d_conv3x3_weights;
    cudaMalloc(&d_conv3x3_weights, conv3x3_weights.size() * sizeof(float));
    cudaMemcpy(d_conv3x3_weights, conv3x3_weights.data(), conv3x3_weights.size() * sizeof(float), cudaMemcpyHostToDevice);
    conv3x3_kernel<<<grid, block>>>(d_concat, d_conv1_out, d_conv3x3_weights, H, W, 2 * reduced_C, 3, conv_out_C);
    cudaDeviceSynchronize();

    // Step 6: Final 1x1 Conv → 6D Pose
    const int final_out_C = 6;
    float* d_final_out;
    cudaMalloc(&d_final_out, final_out_C * H * W * sizeof(float));
    vector<float> conv1x1_final_weights(final_out_C * conv_out_C);
    for (auto& w : conv1x1_final_weights) w = static_cast<float>(rand()) / RAND_MAX;
    float* d_conv1x1_final_weights;
    cudaMalloc(&d_conv1x1_final_weights, final_out_C * conv_out_C * sizeof(float));
    cudaMemcpy(d_conv1x1_final_weights, conv1x1_final_weights.data(), final_out_C * conv_out_C * sizeof(float), cudaMemcpyHostToDevice);
    conv1x1_final_kernel<<<grid, block>>>(d_conv1_out, d_final_out, d_conv1x1_final_weights, H, W, conv_out_C, final_out_C);
    cudaDeviceSynchronize();

    // Step 7: Global Avg Pool → Final Pose Vector
    float* d_pose;
    cudaMalloc(&d_pose, final_out_C * sizeof(float));
    global_avg_pool_kernel<<<1, final_out_C>>>(d_final_out, d_pose, H, W, final_out_C);
    cudaDeviceSynchronize();

    // Step 8: Copy back & Save
    vector<float> pose(final_out_C);
    cudaMemcpy(pose.data(), d_pose, final_out_C * sizeof(float), cudaMemcpyDeviceToHost);
    save_pose_to_csv(pose, output_pose);

    cout << "Predicted 6D Pose saved to: " << output_pose << endl;

    // Cleanup
    cudaFree(d_feat1); cudaFree(d_feat2);
    cudaFree(d_feat1_reduced); cudaFree(d_feat2_reduced);
    cudaFree(d_conv1x1_weights); cudaFree(d_concat);
    cudaFree(d_conv1_out); cudaFree(d_conv3x3_weights);
    cudaFree(d_final_out); cudaFree(d_conv1x1_final_weights);
    cudaFree(d_pose);

    return 0;
}
