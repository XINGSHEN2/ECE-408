#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16

// __constant__ float const_mask[6000];

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int B, const int M, const int C, const int H, const int W, const int K,const int S)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    B - batch_size (number of images in x)
    M - number of output feature maps
    C - number of input feature maps
    H - input height dimension
    W - input width dimension
    K - kernel height and width (K x K)
    S - stride step length
    */

    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;
    // (void)H_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)W_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a
    #define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (C * H * W) + (i2) * (H * W) + (i1) * (W) + i0]
    #define mask_4d(i3, i2, i1, i0) mask[(i3) * (C * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    // #define shared_4d(i2, i1, i0) shared[(i2) * (shared_width * shared_width) + (i1) * (shared_width) + i0]

    // Insert your GPU convolution kernel code here
    int W_grid = ceil(1.0 * W_out / TILE_WIDTH);
    int b = blockIdx.z;
    // int w_base = TILE_WIDTH * (b % W_grid);
    // int h_base = TILE_WIDTH * (b/ W_grid)
    int m = blockIdx.x;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;

    float acc = 0.0f;
    

    for(int c = 0; c < C; c++){
        for (int p = 0; p < K; p++)
            for (int q = 0; q < K; q++)
                acc += in_4d(b, c, h * S + p, w * S + q) * mask_4d(m, c, p, q);
        }
    if(h < H_out && w < W_out){
        out_4d(b, m, h, w) = acc;
    }



    #undef out_4d
    #undef in_4d
    #undef mask_4d
}

	
__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.

    // Useful snippet for error checking
    // cudaError_t error = cudaGetLastError();
    // if(error != cudaSuccess)
    // {
    //     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
    //     exit(-1);
    // }
#define STREAM_NUM 1
    const int H_out = (H - K)/S + 1;
    const int W_out = (W - K)/S + 1;

    float* host_out_temp = (float*) host_output;

    int input_size = B * C * H * W / STREAM_NUM;
    int output_size = B * M * H_out * W_out / STREAM_NUM;
    int mask_size = M * C * K * K;

    int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;
    int Y_grid = H_grid * W_grid;

    dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 gridDim(M, Y_grid, B/STREAM_NUM);

    cudaMalloc((void **) device_input_ptr, B * C * H * W * sizeof(float));
    cudaMalloc((void **) device_output_ptr, B * M * H_out * W_out * sizeof(float));
    cudaMalloc((void **) device_mask_ptr, mask_size * sizeof(float));

    cudaStream_t stream[STREAM_NUM];
    for (int i = 0; i < STREAM_NUM; i++)
        cudaStreamCreate(&stream[i]);

    cudaMemcpyAsync(*device_mask_ptr, host_mask, mask_size * sizeof(float), cudaMemcpyHostToDevice, stream[0]);
    for (int i = 0; i < STREAM_NUM; i++){
        int in_offset = input_size * i;
        int out_offset = output_size * i;
        cudaMemcpyAsync((*device_input_ptr) + in_offset, host_input + in_offset, input_size * sizeof(float), cudaMemcpyHostToDevice, stream[i]);
        conv_forward_kernel<<<gridDim, blockDim, 0, stream[i]>>>((*device_output_ptr) + out_offset, (*device_input_ptr) + in_offset, *device_mask_ptr, B, M, C, H, W, K, S);
        cudaMemcpyAsync(host_out_temp + out_offset, (*device_output_ptr) + out_offset, output_size * sizeof(float), cudaMemcpyDeviceToHost, stream[i]);
    }
    cudaDeviceSynchronize();

    for (int i = 0; i < STREAM_NUM; i++){
        cudaStreamDestroy(stream[i]);
    }

    // Free device memory
    cudaFree(device_input_ptr);
    cudaFree(device_output_ptr);
    cudaFree(device_mask_ptr);
#undef STREAM_NUM
    // cudaMemcpy(*device_input_ptr, host_input, input_size * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(*device_mask_ptr, host_mask, mask_size * sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpyToSymbol(const_mask, host_mask, mask_size * sizeof(float));
}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Set the kernel dimensions and call the kernel
    // const int H_out = (H - K)/S + 1;
    // const int W_out = (W - K)/S + 1;
    // const int shared_width = K + S * (TILE_WIDTH - 1);

    // int H_grid = ceil(1.0 * H_out / TILE_WIDTH);
    // int W_grid = ceil(1.0 * W_out / TILE_WIDTH);

    // int Y_grid = H_grid * W_grid;

    // int shared_men_size = C * shared_width * shared_width * sizeof(float);

    // dim3 blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    // dim3 gridDim(B, M, Y_grid);


    // conv_forward_kernel<<<gridDim, blockDim, shared_men_size>>>(device_output, device_input, device_mask, B, M, C, H, W, K, S);
    return;
    

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
    // Copy the output back to host
    // const int H_out = (H - K)/S + 1;
    // const int W_out = (W - K)/S + 1;

    // int output_size = B * M * H_out * W_out;

    // cudaMemcpy(host_output, device_output, output_size * sizeof(float), cudaMemcpyDeviceToHost);
   
    // // Free device memory
    // cudaFree(device_input);
    // cudaFree(device_output);
    // cudaFree(device_mask);
    return;

}


__host__ void GPUInterface::get_device_properties()
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);

    for(int dev = 0; dev < deviceCount; dev++)
    {
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);

        std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
        std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
        std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
        std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
        std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
        std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
        std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
        std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
        std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
    }
}
