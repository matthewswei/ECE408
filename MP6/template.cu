// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 16

//@@ insert code here
__global__ void floatToChar(float* input, unsigned char* output, int width, int height) {
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x<width && y<height) {
    int ii = blockIdx.z*width*height + y*width + x;
    output[ii] = (unsigned char)(255*input[ii]);
  }
}

__global__ void grayScale(unsigned char* input, unsigned char* output, int width, int height) {
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x<width && y<height) {
    int idx = y*width + x;
    unsigned char r = input[3*idx];
    unsigned char g = input[3*idx + 1];
    unsigned char b = input[3*idx + 2];
    output[idx] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  }
}

__global__ void histogram(unsigned char* input, unsigned int* output, int width, int height) {
  __shared__ unsigned int histogram[HISTOGRAM_LENGTH];
  int idx = threadIdx.x + threadIdx.y*blockDim.x;
  if (idx<HISTOGRAM_LENGTH) {
    histogram[idx] = 0;
  }
  __syncthreads();

  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x<width && y<height) {
    int ii = y*width + x;
    unsigned char val = input[ii];
    atomicAdd(&(histogram[val]), 1);
  }
  __syncthreads();

  if (idx<HISTOGRAM_LENGTH) {
    atomicAdd(&(output[idx]), histogram[idx]);
  }
}

__global__ void CDF(unsigned int* input, float* output, int width, int height) {
  __shared__ unsigned int cdf[HISTOGRAM_LENGTH];
  int x = threadIdx.x;
  cdf[x] = input[x];
  for (unsigned int stride = 1; stride<=HISTOGRAM_LENGTH/2; stride*=2) {
    __syncthreads();
    int idx = 2*stride*(x+1) - 1;
    if (idx<HISTOGRAM_LENGTH) {
      cdf[idx]+=cdf[idx - stride];
    }
  }

  for (int stride = HISTOGRAM_LENGTH/4; stride>0; stride/=2) {
    __syncthreads();
    int idx = 2*stride*(x+1) - 1;
    if ((idx + stride)<HISTOGRAM_LENGTH) {
      cdf[idx + stride]+=cdf[idx];
    }
  }
  __syncthreads();
  output[x] = 1.0*cdf[x]/(width*height);
}

__global__ void histogram_equalization(unsigned char* ucharImage, float* cdf, int width, int height) {
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x<width && y<height) {
    int idx = blockIdx.z*width*height + y*width + x;
    ucharImage[idx] = (unsigned char)min(max(255.0*(cdf[ucharImage[idx]]-cdf[0])/(1.0-cdf[0]), 0.0), 255.0);;
  }
}

__global__ void charToFloat(unsigned char* input, float* output, int width, int height) {
  int x = blockIdx.x*blockDim.x + threadIdx.x;
  int y = blockIdx.y*blockDim.y + threadIdx.y;
  if (x<width && y<height) {
    int ii = blockIdx.z*width*height + y*width + x;
    output[ii] = (float)(input[ii]/255.0);
  }
}

int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  float* deviceImageFloat;
  unsigned char* deviceImageChar;
  unsigned char* deviceImageGrayScale;
  unsigned int* deviceHistogram;
  float* deviceCDF;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostInputImageData = wbImage_getData(inputImage);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  cudaMalloc((void**)&deviceImageFloat, imageHeight*imageWidth*imageChannels*sizeof(float));
  cudaMalloc((void**)&deviceImageChar, imageHeight*imageWidth*imageChannels*sizeof(unsigned char));
  cudaMalloc((void**)&deviceImageGrayScale, imageHeight*imageWidth*sizeof(unsigned char));
  cudaMalloc((void**)&deviceHistogram, HISTOGRAM_LENGTH*sizeof(unsigned int));
  cudaMemset((void*)deviceHistogram, 0, HISTOGRAM_LENGTH*sizeof(unsigned int));
  cudaMalloc((void**)&deviceCDF, HISTOGRAM_LENGTH*sizeof(float));

  cudaMemcpy(deviceImageFloat, hostInputImageData, imageHeight*imageWidth*imageChannels*sizeof(float), cudaMemcpyHostToDevice);

  //@@ Kernel calls
  dim3 dimGrid = dim3(ceil(1.0*imageWidth/BLOCK_SIZE), ceil(1.0*imageHeight/BLOCK_SIZE), imageChannels);
  dim3 dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
  floatToChar<<<dimGrid, dimBlock>>>(deviceImageFloat, deviceImageChar, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  dimGrid = dim3(ceil(1.0*imageWidth/BLOCK_SIZE), ceil(1.0*imageHeight/BLOCK_SIZE), 1);
  grayScale<<<dimGrid, dimBlock>>>(deviceImageChar, deviceImageGrayScale, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  histogram<<<dimGrid, dimBlock>>>(deviceImageGrayScale, deviceHistogram, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  dimGrid = dim3(1, 1, 1);
  dimBlock = dim3(HISTOGRAM_LENGTH, 1, 1);
  CDF<<<dimGrid, dimBlock>>>(deviceHistogram, deviceCDF, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  dimGrid = dim3(ceil(1.0*imageWidth/BLOCK_SIZE), ceil(1.0*imageHeight/BLOCK_SIZE), imageChannels);
  dimBlock = dim3(BLOCK_SIZE, BLOCK_SIZE, 1);
  histogram_equalization<<<dimGrid, dimBlock>>>(deviceImageChar, deviceCDF, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  charToFloat<<<dimGrid, dimBlock>>>(deviceImageChar, deviceImageFloat, imageWidth, imageHeight);
  cudaDeviceSynchronize();

  cudaMemcpy(hostOutputImageData, deviceImageFloat, imageHeight*imageWidth*imageChannels*sizeof(float), cudaMemcpyDeviceToHost);

  wbSolution(args, outputImage);

  //@@ insert code here
  cudaFree(deviceCDF);
  cudaFree(deviceHistogram);
  cudaFree(deviceImageGrayScale);
  cudaFree(deviceImageChar);
  cudaFree(deviceImageFloat);

  return 0;
}
