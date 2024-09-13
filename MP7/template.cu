// Histogram Equalization

#include <wb.h>

#define HISTOGRAM_LENGTH 256

//@@ insert code here

__global__ void toUnsignedChar(float *input, unsigned char *output, int size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < size) {
    output[id] = (unsigned char) (255*input[id]); 
  }
}
__global__ void backtoFloat(unsigned char *input, float *output, int size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < size) {
    output[id] = (float) (input[id]/255.0);
  }
}
__global__ void converttoGray(unsigned char *input, unsigned char *output, int size) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  if (x < size) {
    output[x] = (unsigned char) (0.21*input[3*x] + 0.71*input[3*x+1] + 0.07*input[3*x+2]);
  }
}
__global__ void histogram(unsigned char *input, unsigned int *output, int len) {
  int idx = threadIdx.x + blockDim.x * blockIdx.x;
  __shared__ unsigned int histo_s[HISTOGRAM_LENGTH];

  if (threadIdx.x < HISTOGRAM_LENGTH) {
    histo_s[threadIdx.x] = 0;
  }
  __syncthreads();

  if (idx < len) {
    int pos = input[idx];
    atomicAdd(&(histo_s[pos]), 1);
  }
  __syncthreads();

  if (threadIdx.x < HISTOGRAM_LENGTH) {
    atomicAdd(&(output[threadIdx.x]), histo_s[threadIdx.x]);
  }
}
__global__ void scan(unsigned int *histogram, float *cdf, int size) {
  __shared__ float XY[HISTOGRAM_LENGTH];

  int i = threadIdx.x;
  if (i < HISTOGRAM_LENGTH) XY[i] = histogram[i];
  if (i + blockDim.x < HISTOGRAM_LENGTH) XY[i+blockDim.x] = histogram[i+blockDim.x];

  for (int stride = 1; stride <= blockDim.x; stride *= 2) {
    __syncthreads();
    int index = (i+1) * 2 * stride - 1;
    if (index < HISTOGRAM_LENGTH) {
      XY[index] += XY[index - stride];
    }
  }

  for (int stride = ceil(HISTOGRAM_LENGTH/4.0); stride > 0; stride /= 2) {
    __syncthreads();
    int index = (i+1)*stride*2 - 1;
    if(index + stride < HISTOGRAM_LENGTH) {
      XY[index + stride] += XY[index];
    }
  }
  __syncthreads();
  if (i < HISTOGRAM_LENGTH) cdf[i] = ((float) (XY[i]*1.0)/size);
  if (i + blockDim.x < HISTOGRAM_LENGTH) cdf[i+blockDim.x] = ((float) (XY[i+blockDim.x]*1.0)/size);
}
// histogram equalization function
__global__ void equalize(unsigned char *inout, float *cdf, int size) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < size) {
    float equalized = 255.0*(cdf[inout[id]]-cdf[0])/(1.0-cdf[0]);
    inout[id] = (unsigned char) (min(max(equalized, 0.0), 255.0));
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

  float   *deviceImageFloat;
  unsigned char *deviceImageChar;
  unsigned char *deviceImageCharGrayScale;
  unsigned int  *deviceImageHistogram;
  float   *deviceImageCDF;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  hostInputImageData = wbImage_getData(inputImage);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  hostOutputImageData = wbImage_getData(outputImage);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  // Allocating GPU memory
  cudaMalloc((void **)&deviceImageFloat, imageWidth*imageHeight*imageChannels*sizeof(float));
  cudaMalloc((void **)&deviceImageChar, imageWidth*imageHeight*imageChannels*sizeof(unsigned char));
  cudaMalloc((void **)&deviceImageCharGrayScale, imageWidth*imageHeight*sizeof(unsigned char));
  cudaMalloc((void **)&deviceImageHistogram, HISTOGRAM_LENGTH*sizeof(unsigned int));
  //cudaMemset((void *) deviceImageHistogram, 0, HISTOGRAM_LENGTH *sizeof(unsigned int));
  cudaMalloc((void **)&deviceImageCDF, HISTOGRAM_LENGTH*sizeof(float));
  
  // Copy data to GPU
  cudaMemcpy(deviceImageFloat, hostInputImageData, 
             imageWidth*imageHeight*imageChannels*sizeof(float),cudaMemcpyHostToDevice);
  

  // convert to unsigned char
  dim3 dimGrid1(ceil(imageWidth*imageHeight*imageChannels/512.0), 1, 1);
  dim3 dimBlock1(512,1,1);
  toUnsignedChar<<<dimGrid1,dimBlock1>>>(deviceImageFloat, 
                                         deviceImageChar, imageWidth*imageHeight*imageChannels);
  cudaDeviceSynchronize();
  // convert to gray
  dim3 dimGrid2(ceil(imageWidth*imageHeight/512.0), 1, 1);
  dim3 dimBlock2(512,1,1);
  converttoGray<<<dimGrid2,dimBlock2>>>(deviceImageChar, 
                                 deviceImageCharGrayScale, imageWidth*imageHeight);
  cudaDeviceSynchronize();
  // compute histogram
  dim3 dimGrid3(ceil(imageWidth*imageHeight/256.0), 1, 1);
  dim3 dimBlock3(256,1,1);
  histogram<<<dimGrid3,dimBlock3>>>(deviceImageCharGrayScale, 
                                 deviceImageHistogram, imageWidth*imageHeight);
  cudaDeviceSynchronize();
  // scan
  dim3 dimGrid4(1, 1, 1);
  dim3 dimBloc4(128,1,1);
  scan<<<dimGrid4, dimBloc4>>>(deviceImageHistogram, deviceImageCDF, imageWidth*imageHeight);
  cudaDeviceSynchronize();
  // histogram equalization function
  dim3 dimGrid5(ceil(imageWidth*imageHeight*imageChannels/512.0), 1, 1);
  dim3 dimBlock5(512,1,1);
  equalize<<<dimGrid5,dimBlock5>>>(deviceImageChar, 
                                  deviceImageCDF, imageWidth*imageHeight*imageChannels);
  cudaDeviceSynchronize();
  // cast to float
  dim3 dimGrid6(ceil(imageWidth*imageHeight*imageChannels/512.0), 1, 1);
  dim3 dimBlock6(512,1,1);
  backtoFloat<<<dimGrid6,dimBlock6>>>(deviceImageChar, 
                                 deviceImageFloat, imageWidth*imageHeight*imageChannels);
  cudaDeviceSynchronize();
  cudaMemcpy(hostOutputImageData, deviceImageFloat,
             imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost);


  wbImage_setData(outputImage, hostOutputImageData);
  wbSolution(args, outputImage);

  //@@ insert code here

  cudaFree(deviceImageFloat);
  cudaFree(deviceImageChar);
  cudaFree(deviceImageCharGrayScale);
  cudaFree(deviceImageHistogram);
  cudaFree(deviceImageCDF);

  free(hostInputImageData);
  free(hostOutputImageData);

  return 0;
}
