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
#define MASK_WIDTH 3
#define TILE_WIDTH 8
//@@ Define constant memory for device kernel here
__constant__ float deviceKernel[MASK_WIDTH][MASK_WIDTH][MASK_WIDTH];

__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  //@@ Insert kernel code here
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;

  __shared__ float tile[TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1][TILE_WIDTH + MASK_WIDTH - 1];

  int x_out = blockIdx.x * TILE_WIDTH + tx;
  int y_out = blockIdx.y * TILE_WIDTH + ty;
  int z_out = blockIdx.z * TILE_WIDTH + tz;

  // int x_in = x_out - MASK_WIDTH / 2;
  // int y_in = y_out - MASK_WIDTH / 2;
  // int z_in = z_out - MASK_WIDTH / 2;

  int x_in = x_out - 1;
  int y_in = y_out - 1;
  int z_in = z_out - 1;

  float Pvalue = 0.0f;
  if((x_in >= 0) && (y_in >= 0) && (z_in >= 0) && (x_in < x_size) && (y_in < y_size) && (z_in < z_size)){
    tile[tz][ty][tx] = input[z_in * x_size * y_size + y_in * x_size + x_in];
  } else {
    tile[tz][ty][tx] = 0.0f;
  }
  __syncthreads();

  if(tx < TILE_WIDTH && ty < TILE_WIDTH && tz < TILE_WIDTH){
    for(int i = 0; i < MASK_WIDTH; i++){
      for(int j = 0; j < MASK_WIDTH; j++){
        for(int k = 0; k < MASK_WIDTH; k++){
          Pvalue += deviceKernel[i][j][k] * tile[i + tz][j + ty][k + tx];
        }
      }
    }
    if(x_out < x_size && y_out < y_size && z_out < z_size){
      output[z_out * x_size * y_size + y_out * x_size + x_out] = Pvalue;
    }
  }

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
  // float *deviceKernel;
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

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  cudaMalloc((void**) &deviceInput, (inputLength - 3) * sizeof(float));
  cudaMalloc((void**) &deviceOutput, (inputLength - 3) * sizeof(float));

  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  cudaMemcpy(deviceInput, hostInput + 3, (inputLength - 3) * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(deviceKernel, hostKernel, kernelLength * sizeof(float));
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  dim3 DimBlock(TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1, TILE_WIDTH + MASK_WIDTH - 1);
  dim3 DimGrid(ceil(x_size / (1.0 * TILE_WIDTH)), ceil(y_size / (1.0 * TILE_WIDTH)), ceil(z_size / (1.0 * TILE_WIDTH)));
  // dim3 DimGrid((x_size - 1 + TILE_WIDTH) / TILE_WIDTH, (y_size - 1 + TILE_WIDTH) / TILE_WIDTH, (z_size - 1 + TILE_WIDTH) / TILE_WIDTH);
  //@@ Launch the GPU kernel here

  conv3d<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  cudaMemcpy(hostOutput + 3, deviceOutput, (inputLength - 3) * sizeof(float), cudaMemcpyDeviceToHost);
  wbTime_stop(Copy, "Copying data from the GPU");

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

