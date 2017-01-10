// Corresponding header file: /include/mirror_ops.h
#include <cuda_runtime.h>
#include <stdio.h>    

/* Mirror operations */

__global__ 
void mirror(const uchar4* const inputChannel, uchar4* outputChannel, int numRows, int numCols, bool vertical)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if ( col >= numCols || row >= numRows )
  {
   return;
  }

  if(!vertical)
  { 
  
    int thread_x = blockDim.x * blockIdx.x + threadIdx.x;
    int thread_y = blockDim.y * blockIdx.y + threadIdx.y;
    
    int thread_x_new = thread_x;
    int thread_y_new = numRows-thread_y;

    int myId = thread_y * numCols + thread_x;
    int myId_new = thread_y_new * numCols + thread_x_new;
    outputChannel[myId_new] = inputChannel[myId];
   	
  }

  else
  {
  	int thread_x = blockDim.x * blockIdx.x + threadIdx.x;
    int thread_y = blockDim.y * blockIdx.y + threadIdx.y;
    
    int thread_x_new = numCols-thread_x;
    int thread_y_new = thread_y;

    int myId = thread_y * numCols + thread_x;
    int myId_new = thread_y_new * numCols + thread_x_new;
  
  	outputChannel[myId_new] = inputChannel[myId];  // linear data store in global memory	
  }
}         



uchar4* mirror_ops(const uchar4* const h_in, size_t numRows, size_t numCols, bool vertical)
{
	//Set reasonable block size (i.e., number of threads per block)
	const dim3 blockSize(4,4,1);
  //Calculate Grid SIze
  int a=numCols/blockSize.x, b=numRows/blockSize.y;	
  const dim3 gridSize(a+1,b+1,1);

  const size_t numPixels = numRows * numCols;

  cudaMalloc(d_outputImageRGBA, sizeof(uchar4) * numPixels);

  //Call mirror kernel.
  mirror<<<gridSize, blockSize>>>(d_inputImageRGBA, d_outputImageRGBA, numRows, numCols, vertical);

  cudaDeviceSynchronize(); 
 
  //Initialize memory on host for output uchar4*
  uchar4* h_out;
  h_out = (uchar4*)malloc(sizeof(uchar4) * numPixels)

  //copy output from device to host
  cudaMemcpy(h_out, d_outputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost);
  
  //cleanup memory on device
  cudaFree(d_inputImageRGBA);
  cudaFree(d_outputImageRGBA);

  //return h_out
	return h_out;
}