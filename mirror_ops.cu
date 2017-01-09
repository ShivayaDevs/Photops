// Corresponding header file: /include/mirror_ops.h
#include <cuda_runtime.h>
#include <stdio.h>    

/* Mirror operations */

__global__ 
void mirror(const uchar4* const inputChannel, uchar4* outputChannel, int numRows, int numCols, bool vertical)
{
  __shared__ uchar4 sharedBlockA[4][4];   // 1. shared memory for reverse swap
  __shared__ uchar4 sharedBlockB[4][4];   // 2. shared memory for reverse swap

  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if ( col >= numCols || row >= numRows )
  {
   return;
  }
  
  int tx = threadIdx.x;   // thread index X-Dir
  int ty = threadIdx.y;   // thread index Y-Dir
  
  int dX = numCols;	//  the pictures width
  int dY = numRows;	//  the pictures height

  if(vertical)
  {
  	int blockIdxA = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * dX;	 //  begin read
  	int blockIdxB = dX * dY - blockIdxA - blockDim.y * dX - blockDim.x;		 //  store data
  
  	sharedBlockA[ty][tx].x = inputChannel[blockIdxA + ty * dX + tx].x;  // linear data fetch from global memory
  	sharedBlockA[ty][tx].y = inputChannel[blockIdxA + ty * dX + tx].y; 
  	sharedBlockA[ty][tx].z = inputChannel[blockIdxA + ty * dX + tx].z; 
  	
		__syncthreads ();   // wait for all threads to reach this point

		sharedBlockB[ty][tx].x = sharedBlockA[3-ty][3-tx].x; // mirror each element in the cache
		sharedBlockB[ty][tx].y = sharedBlockA[3-ty][3-tx].y;
		sharedBlockB[ty][tx].z = sharedBlockA[3-ty][3-tx].z;
		
		__syncthreads();   // wait for all threads to reach this point
		
		unsigned char red   = 	sharedBlockB[ty][tx].x;
		unsigned char blue   = 	sharedBlockB[ty][tx].y;
		unsigned char green   = 	sharedBlockB[ty][tx].z;
		outputChannel[blockIdxB + ty * dX + tx] = make_uchar4(red,blue,green,255);   // linear data store in global memory
   	
  }
	else
  {
  	int blockIdxA = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * dX;	 //  begin read
  	int blockIdxB = blockIdx.y * blockDim.y * dX + dX*blockDim.y - blockIdx.x*blockDim.x - blockDim.x*blockDim.y; //  store data
  
  	sharedBlockA[ty][tx].x = inputChannel[blockIdxA + ty * dX + tx].x;  // linear data fetch from global memory
  	sharedBlockA[ty][tx].y = inputChannel[blockIdxA + ty * dX + tx].y; 
  	sharedBlockA[ty][tx].z = inputChannel[blockIdxA + ty * dX + tx].z; 
 
		__syncthreads ();   // wait for all threads to reach this point

		sharedBlockB[ty][tx].x = sharedBlockA[ty][3-tx].x; // mirror each element in the cache
		sharedBlockB[ty][tx].y = sharedBlockA[ty][3-tx].y;
		sharedBlockB[ty][tx].z = sharedBlockA[ty][3-tx].z;
	
    __syncthreads();   // wait for all threads to reach this point
		
		unsigned char red   = 	sharedBlockB[ty][tx].x;
		unsigned char blue   = 	sharedBlockB[ty][tx].y;
		unsigned char green   = 	sharedBlockB[ty][tx].z;

   	outputChannel[blockIdxB + ty * dX + tx] = make_uchar4(red,blue,green,255);   // linear data store in global memory
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