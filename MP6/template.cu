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

__global__ void scan(float *input, float *output, int len, float *sum_array) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this
  //@@ function and call them from the host
  __shared__ float T[BLOCK_SIZE * 2];
  int start = blockIdx.x * blockDim.x * 2 + threadIdx.x * 2;
  if (start < len)
    T[threadIdx.x * 2] = input[start];
  else
    T[threadIdx.x * 2] = 0.0f;
  if (start + 1 < len)
    T[threadIdx.x * 2 + 1] = input[start + 1];
  else
    T[threadIdx.x * 2 + 1] = 0.0f;
  int stride = 1;
  while (stride < BLOCK_SIZE * 2){
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 -1;
    if(index < 2 * BLOCK_SIZE && (index - stride) >= 0){
      T[index] += T[index - stride];
    }
    stride = stride * 2;
  } 
  stride = BLOCK_SIZE / 2;
  while (stride > 0){
    __syncthreads();
    int index = (threadIdx.x + 1) * stride * 2 - 1;
    if (index + stride < BLOCK_SIZE * 2)
      T[index + stride] += T[index];
    stride = stride / 2;
  }
  __syncthreads();
  if (start < len){
    output[start] = T[threadIdx.x * 2];}
  if (start + 1 < len){
    output[start + 1] = T[threadIdx.x * 2 + 1];}
  if (threadIdx.x == BLOCK_SIZE - 1 && (start + 1 < len))
    sum_array[blockIdx.x] = T[threadIdx.x * 2 + 1];
}

__global__ void scan_add(float *output, float *sum_array, int len) {
  int start = blockIdx.x * blockDim.x * 2 + threadIdx.x * 2;
  if (blockIdx.x > 0 && (start < len)){
    output[start] += sum_array[blockIdx.x - 1];
  }
  if (blockIdx.x > 0 && (start + 1 < len)){
    output[start + 1] += sum_array[blockIdx.x - 1];
  }
}
int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list
  float *sum_array_1; // the sum of each block
  float *sum_array_2;
  int numSumarray;

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
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

  //@@ Initialize the grid and block dimensions here
  numSumarray = (numElements - 1) / (2 * BLOCK_SIZE) + 1;
  cudaMalloc((void **) &sum_array_1, numSumarray * sizeof(float));
  cudaMalloc((void **) &sum_array_2, numSumarray * sizeof(float));
  dim3 DimGrid(numSumarray, 1, 1);
  dim3 DimBlock(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
  scan<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, numElements, sum_array_1);
  scan<<<dim3 (1, 1, 1), DimBlock>>>(sum_array_1, sum_array_2, numSumarray, NULL);
  scan_add<<<DimGrid, DimBlock>>>(deviceOutput, sum_array_2, numElements);

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceOutput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");
  // for (int i = 1; i < (numElements - 1) / (2 * BLOCK_SIZE) + 1; i++){
  //   for (int j = 0; j < BLOCK_SIZE * 2; j++){
  //     if (i * BLOCK_SIZE * 2 + j < numElements){
  //       hostOutput[i * BLOCK_SIZE * 2 + j] += hostOutput[i * BLOCK_SIZE * 2 - 1];
  //     }
  //   }
  // }
  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
