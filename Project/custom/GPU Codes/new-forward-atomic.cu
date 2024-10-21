#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

#define TILE_WIDTH 16

__constant__ float Mask[6000];

__global__ void conv_forward_kernel(float *output, const float *input, const float *mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    /*
    Modify this function to implement the forward pass described in Chapter 16.
    We have added an additional dimension to the tensors to support an entire mini-batch
    The goal here is to be correct AND fast.

    Function paramter definitions:
    output - output
    input - input
    mask - convolution kernel
    Batch - batch_size (number of images in x)
    Map_out - number of output feature maps
    Channel - number of input feature maps
    Height - input height dimension
    Width - input width dimension
    K - kernel height and width (K x K)
    */

    extern __shared__ float subTile[];

    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    const int Width_share = TILE_WIDTH + K - 1;
    // (void)Height_out; // silence declared but never referenced warning. remove this line when you start working
    // (void)Width_out; // silence declared but never referenced warning. remove this line when you start working

    // We have some nice #defs for you below to simplify indexing. Feel free to use them, or create your own.
    // An example use of these macros:
    // float a = in_4d(0,0,0,0)
    // out_4d(0,0,0,0) = a

    #define out_4d(i3, i2, i1, i0) output[(i3) * (Map_out * Height_out * Width_out) + (i2) * (Height_out * Width_out) + (i1) * (Width_out) + i0]
    #define in_4d(i3, i2, i1, i0) input[(i3) * (Channel * Height * Width) + (i2) * (Height * Width) + (i1) * (Width) + i0]
    #define mask_4d(i3, i2, i1, i0) Mask[(i3) * (Channel * K * K) + (i2) * (K * K) + (i1) * (K) + i0]
    #define subtile_4d(i2, i1, i0) subTile[(i2) * (Width_share * Width_share) + (i1) * Width_share + i0]

    // Insert your GPU convolution kernel code here
    int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;
    int bx = blockIdx.x; int by = blockIdx.y; int bz = blockIdx.z;

    int w_grid = (Width_out-1)/TILE_WIDTH + 1;
    int b = bz;
    int m = bx;
    int c = tz;
    int w = (by%w_grid)*TILE_WIDTH + tx;
    int h = (by/w_grid)*TILE_WIDTH + ty;
    int subtile_w = (by%w_grid)*TILE_WIDTH;
    int subtile_h = (by/w_grid)*TILE_WIDTH;

    for(int i = ty; i<Width_share; i+=TILE_WIDTH) {
        for(int j = tx; j<Width_share; j+=TILE_WIDTH) {
            if (subtile_h + i < Height && subtile_w + j < Width) {
                subtile_4d(c, i, j) = in_4d(b, c, subtile_h + i, subtile_w + j);
            }
            
        }
    }

    __syncthreads();

    if (h<Height_out && w<Width_out) {
        float result = 0;
        for (int p = 0; p<K; p++) {
            for (int q = 0; q<K; q++) {
                result += subtile_4d(c, ty + p, tx + q) * mask_4d(m, c, p, q);
            }
        }
        atomicAdd((&out_4d(b, m, h, w)), result);
    }       

    #undef out_4d
    #undef in_4d
    #undef mask_4d
    #undef subtile_4d
}


__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Allocate memory and copy over the relevant data structures to the GPU

    // We pass double pointers for you to initialize the relevant device pointers,
    //  which are passed to the other two functions.
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;
    cudaMalloc((void**) device_input_ptr, Batch*Channel*Height*Width*sizeof(float));
    cudaMalloc((void**) device_output_ptr, Batch*Map_out*Height_out*Width_out*sizeof(float));
    // cudaMalloc((void**) device_mask_ptr, Map_out*Channel*K*K*sizeof(float));
    cudaMemcpy(*device_input_ptr, host_input, Batch*Channel*Height*Width*sizeof(float), cudaMemcpyHostToDevice);
    // cudaMemcpy(*device_mask_ptr, host_mask, Map_out*Channel*K*K*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(Mask, host_mask, Map_out*Channel*K*K*sizeof(float));

    // Useful snippet for error checking
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess)
    {
        std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
        exit(-1);
    }

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    // Set the kernel dimensions and call the kernel
    int Height_out = Height - K + 1;
    int Width_out = Width - K + 1;
    int Tile_out = TILE_WIDTH + K - 1;

    dim3 dimGrid(Map_out, ceil(1.0*Height_out/TILE_WIDTH )*ceil(1.0*Width_out/TILE_WIDTH ), Batch);
    dim3 dimBlock(TILE_WIDTH , TILE_WIDTH , Channel);

    conv_forward_kernel<<<dimGrid, dimBlock, Channel*Tile_out*Tile_out*sizeof(float)>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);
    //conv_forward_kernel<<<dimGrid, dimBlock>>>(device_output, device_input, device_mask, Batch, Map_out, Channel, Height, Width, K);

}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int Batch, const int Map_out, const int Channel, const int Height, const int Width, const int K)
{
    const int Height_out = Height - K + 1;
    const int Width_out = Width - K + 1;

    // Copy the output back to host
    cudaMemcpy(host_output, device_output, Batch*Map_out*Height_out*Width_out*sizeof(float), cudaMemcpyDeviceToHost);

    // Free device memory
    // cudaFree(device_mask);
    cudaFree(device_output);
    cudaFree(device_input);
    
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