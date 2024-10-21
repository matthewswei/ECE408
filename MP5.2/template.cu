// MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len, int scanSum) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float subtile[2*BLOCK_SIZE];

  /* Load elements from global */
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int start, stride, index;
  
  if (scanSum==0) {
    start = 2*bx*BLOCK_SIZE + tx;
    stride = BLOCK_SIZE;
  } else {
    start = 2*BLOCK_SIZE*(tx+1) - 1;
    stride = 2*BLOCK_SIZE;
  }
  
  if (start<len) {  
    subtile[tx] = input[start]; 
  } else { 
    subtile[tx] = 0; 
  }
  if ((start + stride)<len) {  
    subtile[BLOCK_SIZE+tx] = input[start + stride]; 
  } else { 
    subtile[BLOCK_SIZE+tx] = 0; 
  }
  __syncthreads();

  stride = 1;
  while (stride<2*BLOCK_SIZE) {
    index = (tx+1)*stride*2 - 1;
    if ((index<2*BLOCK_SIZE)&&((index-stride)>=0)) {
      subtile[index]+=subtile[index-stride];
    }
    __syncthreads();
    stride*=2;
  }

  stride = BLOCK_SIZE/2;
  while (stride>0) {
    index = (tx+1)*stride*2 - 1;
    if ((index + stride)<2*BLOCK_SIZE) {
      subtile[index + stride] += subtile[index];
    }
    __syncthreads();
    stride/=2;
  }
  
  start = 2*bx*BLOCK_SIZE + tx;
  if (start<len) {
    output[start] = subtile[tx];
  }
  if ((start + BLOCK_SIZE)<len) {
    output[start + BLOCK_SIZE] = subtile[tx + BLOCK_SIZE];
  }
}

__global__ void add(float *input, float *output, float *sum, int len) {
  int tx = threadIdx.x;
  int bx = blockIdx.x;
  int start = 2*bx*BLOCK_SIZE + tx;

  __shared__ float addsum;

  if (tx==0) {
    if (bx == 0) { 
      addsum = 0; 
    } else { 
      addsum = sum[bx - 1]; 
    }
  }
  __syncthreads();

  if (start<len) {
    output[start] = input[start] + addsum;
  }

  if ((start + BLOCK_SIZE)<len) {
    output[start + BLOCK_SIZE] = input[start + BLOCK_SIZE]  + addsum;
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  float *deviceScanBlock;
  float *deviceScanSum;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceScanBlock, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceScanSum, 2 * BLOCK_SIZE * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  dim3 dimGrid(ceil(numElements/(2.0*BLOCK_SIZE)), 1, 1);
  dim3 dimBlock(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<dimGrid, dimBlock>>>(deviceInput, deviceScanBlock, numElements, 0);
  cudaDeviceSynchronize();

  scan<<<1, dimBlock>>>(deviceScanBlock, deviceScanSum, numElements, 1);
  cudaDeviceSynchronize();

  add<<<dimGrid, dimBlock>>>(deviceScanBlock, deviceOutput, deviceScanSum, numElements);
  cudaDeviceSynchronize();

  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  cudaFree(deviceScanBlock);
  cudaFree(deviceScanSum);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
