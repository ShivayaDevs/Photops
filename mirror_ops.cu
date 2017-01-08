// Corresponding header file: /include/mirror_ops.h
#include <cuda_runtime.h>
#include <stdio.h>    
#include <helper_cuda.h>
#include <helper_functions.h>   

/* Write the code to mirror the image.
   Mirror's orientation can be both horizontal as well as vertical.
   Decide the parameters for yourself and return a pointer to the new image.
   Or maybe, you can deallocate the memory of the incoming image after the operation.

   You will receive a pointer to h_in so tasks like allocating memory 
   to GPU - you need to handle them yourself.  
*/

unsigned char *d_red, *d_green, *d_blue;

__global__
void mirror(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols, bool vertical)
{

  __shared__ unsigned char sharedBlockA[4][4];   // 1. shared memory for reverse swap

  __shared__ unsigned char sharedBlockB[4][4];   // 2. shared memory for reverse swap

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
  
  	sharedBlockA[ty][tx] = inputChannel[blockIdxA + ty * dX + tx];  // linear data fetch from global memory
		__syncthreads ();   // wait for all threads to reach this point
		sharedBlockB[ty][tx] = sharedBlockA[3-ty][3-tx]; // mirror each element in the cache
		__syncthreads();   // wait for all threads to reach this point

   	outputChannel[blockIdxB + ty * dX + tx] = sharedBlockB[ty][tx];   // linear data store in global memory
  }

  else
  {
  	int blockIdxA = blockIdx.x * blockDim.x + blockIdx.y * blockDim.y * dX;	 //  begin read
  	int blockIdxB = blockIdx.y * blockDim.y * dX + dX*blockDim.y - blockIdx.x*blockDim.x - blockDim.x*blockDim.y; //  store data
  
  	sharedBlockA[ty][tx] = inputChannel[blockIdxA + ty * dX + tx];  // linear data fetch from global memory
		__syncthreads ();   // wait for all threads to reach this point
		sharedBlockB[ty][tx] = sharedBlockA[ty][3-tx]; // mirror each element in the cache
		__syncthreads();   // wait for all threads to reach this point

   	outputChannel[blockIdxB + ty * dX + tx] = sharedBlockB[ty][tx];   // linear data store in global memory
  }

}


__global__
void separateChannels(const uchar4* const inputImageRGBA,
                      int numRows,
                      int numCols,
                      unsigned char* const redChannel,
                      unsigned char* const greenChannel,
                      unsigned char* const blueChannel)
{

  
  int absolute_image_position_x = blockDim.x * blockIdx.x + threadIdx.x;
  int absolute_image_position_y = blockDim.y * blockIdx.y + threadIdx.y;

  if ( absolute_image_position_x >= numCols ||
      absolute_image_position_y >= numRows )
  {
       return;
  }
  
  int thread_1D_pos = absolute_image_position_y * numCols + absolute_image_position_x;

  redChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].x;
  greenChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].y;
  blueChannel[thread_1D_pos] = inputImageRGBA[thread_1D_pos].z;

}

__global__
void recombineChannels(const unsigned char* const redChannel,
                       const unsigned char* const greenChannel,
                       const unsigned char* const blueChannel,
                       uchar4* const outputImageRGBA,
                       int numRows,
                       int numCols)
{
  const int2 thread_2D_pos = make_int2( blockIdx.x * blockDim.x + threadIdx.x,
                                        blockIdx.y * blockDim.y + threadIdx.y);

  const int thread_1D_pos = thread_2D_pos.y * numCols + thread_2D_pos.x;

  if (thread_2D_pos.x >= numCols || thread_2D_pos.y >= numRows)
    return;

  unsigned char red   = redChannel[thread_1D_pos];
  unsigned char green = greenChannel[thread_1D_pos];
  unsigned char blue  = blueChannel[thread_1D_pos];

  //Alpha should be 255 for no transparency
  uchar4 outputPixel = make_uchar4(red, green, blue, 255);

  outputImageRGBA[thread_1D_pos] = outputPixel;
}




void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage)
{

  //allocate memory for the three different channels
  //original
  cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage);
  cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage);
  cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage);

}

void cleanup() {
  cudaFree(d_red);
  cudaFree(d_green);
  cudaFree(d_blue);
}

uchar4* mirror_ops(const uchar4* const h_in, size_t numRows, size_t numCols, bool vertical)
{
	//Set reasonable block size (i.e., number of threads per block)

  const dim3 blockSize(4,4,1);
  //Calculate Grid SIze
  int a=numCols/blockSize.x, b=numRows/blockSize.y;	
  const dim3 gridSize(a+1,b+1,1);

  const size_t numPixels = numRows * numCols;

  //allocate memory on the device for both input and output
  cudaMalloc(d_inputImageRGBA, sizeof(uchar4) * numPixels);
  cudaMalloc(d_outputImageRGBA, sizeof(uchar4) * numPixels);
  cudaMemset(*d_outputImageRGBA, 0, numPixels * sizeof(uchar4)); //make sure no memory is left laying around

  //copy input array to the GPU
  cudaMemcpy(d_inputImageRGBA, h_in, sizeof(uchar4) * numPixels, cudaMemcpyHostToDevice);

  //blurred
  cudaMalloc(d_redBlurred, sizeof(unsigned char) * numPixels);
  cudaMalloc(d_greenBlurred,  sizeof(unsigned char) * numPixels);
  cudaMalloc(d_blueBlurred,   sizeof(unsigned char) * numPixels);
  cudaMemset(*d_redBlurred,   0, sizeof(unsigned char) * numPixels);
  cudaMemset(*d_greenBlurred, 0, sizeof(unsigned char) * numPixels);
  cudaMemset(*d_blueBlurred,  0, sizeof(unsigned char) * numPixels);

  allocateMemoryAndCopyToGPU(numRows, numCols);

  //Launch a kernel for separating the RGBA image into different color channels
  separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA, numRows, numCols, d_red,d_green, d_blue);

  cudaDeviceSynchronize(); 

  //Call mirror kernel here 3 times, once for each color channel.
  mirror<<<gridSize, blockSize>>>(d_red, d_redBlurred, numRows, numCols, vertical);
  mirror<<<gridSize, blockSize>>>(d_green, d_greenBlurred, numRows, numCols, vertical);
  mirror<<<gridSize, blockSize>>>(d_blue, d_blueBlurred, numRows, numCols, vertical);


  cudaDeviceSynchronize(); 

  //Now we recombine the results.

  recombineChannels<<<gridSize, blockSize>>>(d_redBlurred,
                                             d_greenBlurred,
                                             d_blueBlurred,
                                             d_outputImageRGBA,
                                             numRows,
                                             numCols);
  cudaDeviceSynchronize(); 

  //cleanup memory
  cleanup();
  cudaFree(d_redBlurred);
  cudaFree(d_greenBlurred);
  cudaFree(d_blueBlurred);

  cudaDeviceSynchronize(); 

  //Initialize memory on host for output uchar4*
  uchar4* h_out;
  h_out = (uchar4*)malloc(sizeof(uchar4) * numPixels)

  //copy output from device to host
  cudaMemcpy(h_out, d_outputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize();

  //cleanup memory on device
  cudaFree(d_inputImageRGBA);
  cudaFree(d_outputImageRGBA);

  //return h_out
	return h_out;
}