#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here
#define TILE_SIZE 8
#define KERNEL_SIZE 3

//@@ Define constant memory for device kernel here
__constant__ float deviceKernel[KERNEL_SIZE*KERNEL_SIZE*KERNEL_SIZE];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {

  //@@ Insert kernel code here
  __shared__ float Cache[10][10][10];

  // Get block and thread values
  int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;
  int bx = blockIdx.x; int by = blockIdx.y; int bz = blockIdx.z;

  // Initialize x, y, z with values and offset values for x, y, z for convolution
  int x = bx * TILE_SIZE + tx;
  int y = by * TILE_SIZE + ty;
  int z = bz * TILE_SIZE + tz;
  int x_mult = x - (KERNEL_SIZE/2);
  int y_mult = y - (KERNEL_SIZE/2);
  int z_mult = z - (KERNEL_SIZE/2);

  // Set cache will values of the offsetted matrix
  if ((0<=x_mult) && (x_mult<x_size) && (0<=y_mult) && (y_mult<y_size) && (0<=z_mult) && (z_mult<z_size)) {
    Cache[tz][ty][tx] = input[(z_mult*(y_size*x_size)) + (y_mult*x_size) + x_mult];
  } else {
    Cache[tz][ty][tx] = 0;
  }

  __syncthreads();

  // Do the 3D convolution and store value in output
  float Pvalue = 0;
    if((tx<TILE_SIZE) && (tx>=0) && (ty<TILE_SIZE) && (ty>=0) && (tz<TILE_SIZE) && (tz>=0)) {
      for(int i = 0; i<KERNEL_SIZE; i++){
        for(int j = 0; j<KERNEL_SIZE; j++){
          for(int k = 0; k<KERNEL_SIZE; k++){
            float temp = deviceKernel[(k*(KERNEL_SIZE*KERNEL_SIZE)) + (j*KERNEL_SIZE) + i];
            temp *= Cache[tz + k][ty + j][tx + i];
            Pvalue+=temp;
          }
        }
      }
      if((z<z_size) && (z>=0) && (y<y_size) && (y>=0) &&(x<x_size) && (x>=0)) {
        output[(z*(y_size*x_size)) + (y*x_size) + x] = Pvalue;
      }
    }
    __syncthreads();

}

int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  int sizeM = (inputLength-3)*sizeof(float);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void**) &deviceInput, sizeM);
  cudaMalloc((void**) &deviceOutput, sizeM);
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput+3, sizeM, cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(deviceKernel, hostKernel, kernelLength*sizeof(float));
  wbTime_stop(Copy, "Copying data to the GPU");


  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 dimGrid(ceil(x_size/8.0), ceil(y_size/8.0), ceil(z_size/8.0));
  dim3 dimBlock(10, 10, 10);
  //@@ Launch the GPU kernel here
  conv3d<<<dimGrid, dimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)

  wbTime_stop(Copy, "Copying data from the GPU");
  cudaMemcpy(hostOutput + 3, deviceOutput, sizeM, cudaMemcpyDeviceToHost);
  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}