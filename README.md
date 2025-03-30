##OPTIMING LIDAR MONOCULAR ODOMETRY PIPLINE USING CUDA PARALELLIZATION

Existing visual odometry systems often suffer from significant computational overhead, limiting their ability to perform real-time processing, especially when combining Lidar and monocular data. They typically rely on CPU-bound implementations, leading to bottlenecks in feature extraction, vector transformations, and pose estimation, particularly when handling large-scale environments or high-resolution input streams. To address these limitations, we parallelized critical components of the pipeline using CUDA, focusing on optimizing convolution operations, memory access patterns, shared memory and pinned memory utilization within the ResNet encoder, RectifyNet, and PoseNet modules. Our CUDA-based implementation achieves considerable speedup compared to traditional approaches, while maintaining or improving the accuracy of motion estimation and depth reconstruction. This work demonstrates the effectiveness of GPU-accelerated techniques in overcoming the computational constraints of conventional VO systems, paving the way for their deployment in resource-constrained or real-time robotics scenarios. 
