// Corresponding header file: /include/square_ops.h
#include <cuda_runtime.h>
#include "include/blur_ops.h"

/* Write the code to square(blur) the image.
   2 cases as already specified.

   Decide the parameters for yourself and return a pointer to the new image.
   Or maybe, you can deallocate the memory of the incoming image after the operation.

   You will receive a pointer to h_inputImage so tasks like allocating memory 
   to GPU - you need to handle them yourself.  
*/


__global__ 
void square(const uchar4* d_in, uchar4* d_sq, size_t numRows, size_t numCols, size_t n_numRows, size_t n_numCols, uchar4 color)
{
 	int y = blockDim.x*blockIdx.x + threadIdx.x;	//column
	int x = blockDim.y*blockIdx.y + threadIdx.y;	//row
	int index = x*numRows + y;										//previous index of pixel
	int n_index = x*n_numRows + y;								//new index of pixel

	if(y >= n_numCols || x >= n_numRows)  				//check out of bound
	  return;

	if(y < numCols && x < numRows)								
	  d_sq[n_index] = d_in[index];
	else
	  d_sq[n_index] = color;
}

__global__ 
void square_blur(const uchar4* d_in, uchar4* d_sq, const float* const d_filter, const int filterWidth, 
                 size_t numRows, size_t numCols,  size_t n_numRows, size_t n_numCols)
{
	int y = blockDim.x*blockIdx.x + threadIdx.x;  //column
	int x = blockDim.y*blockIdx.y + threadIdx.y;	//row
	int index = x*numRows + y;										//previous index of pixel
	int n_index = x*n_numRows + y;						//new index of pixel

	if(y >= n_numCols || x >= n_numRows)  		//check out of bound
	  return;

	if(y < numCols && x < numRows)								
	  d_sq[n_index] = d_in[index];
	else
	{
	  int prev_x = x - (n_numRows - numRows);		//finding pixel to blurr
	  int prev_y = y - (n_numCols - numCols);
	  int prev_index = prev_x * numRows + prev_y;

	  uchar4 sum = make_uchar4(0,0,0,225);
    for(int px = 0; px < filterWidth; px++)		//calculating new pixel intensity
	  {
	    for(int py = 0; py < filterWidth; py++)
	    {
       	int row = x + px - (filterWidth/2);
        int col = y + py - (filterWidth/2);
        row = min( max(0,row), static_cast<unsigned int>(numCols-1));
        col = min( max(0,col), static_cast<unsigned int>(numRows-1));
        sum.x+= d_filter[py*filterWidth+px] * ( static_cast<float>( d_in[prev_index].x ) );
        sum.y+= d_filter[py*filterWidth+px] * ( static_cast<float>( d_in[prev_index].y ) );
        sum.z+= d_filter[py*filterWidth+px] * ( static_cast<float>( d_in[prev_index].z ) );
	    }
  	}

  	d_sq[n_index] = sum;
	}
}

/* 
	n_numRows and n_numCols are the new row and column sizes
	d_sq represents output image intensities
*/

uchar4* square(uchar4* const d_image, size_t numRows, size_t numCols, size_t &n_numRows, size_t &n_numCols, uchar4 color)
{
	size_t newSize;
  const dim3 blockSize(64, 64, 1);  
  const dim3 gridSize(numRows/blockSize.x+1, numCols/blockSize.y+1,1);  
   
  if(numCols > numRows)		//setting new cols and rows size
  {
    n_numRows = numCols;
    n_numCols = numCols;
  }
  else
  {
    n_numCols = numRows; 
    n_numRows = numRows;
  }
  
	newSize = n_numRows * n_numCols;
  uchar4* d_sq;
  cudaMalloc(&d_sq, sizeof(uchar4)*newSize);
	square<<<gridSize, blockSize>>>(d_image, d_sq, numRows, numCols, n_numRows, n_numCols, color);

  uchar4 *h_out = new uchar4[n_numRows * n_numCols * sizeof(uchar4)];
  cudaMemcpy(h_out, d_sq, n_numRows * n_numCols * sizeof(uchar4), cudaMemcpyDeviceToHost);
  return h_out; 
}


uchar4* square_blur(uchar4* const d_image, size_t numRows, size_t numCols, size_t &n_numRows, size_t &n_numCols, int blurKernelWidth, float blurKernelSigma)
{
	size_t newSize;
  const dim3 blockSize(64, 64, 1);  
  const dim3 gridSize(numRows/blockSize.x+1, numCols/blockSize.y+1,1);  
  
  if(numCols > numRows)		//setting new cols and rows size
  {
    n_numRows = numCols;
    n_numCols = numCols;
  }
  else
  {
    n_numCols = numRows; 
    n_numRows = numRows;
  }
  
  newSize = n_numRows * n_numCols;
  uchar4* d_sq;
  cudaMalloc(&d_sq, sizeof(uchar4)*newSize);
  
  /*setting the filter: we need to change blurkernelWidth and blurKernelSigma to change the filter */
  float* h_filter;
  size_t filterWidth;
  // setFilter(&h_filter, &filterWidth, blurKernelWidth, blurKernelSigma);

  float *d_filter; 	//creating device copy of h_filter
  cudaMalloc(&d_filter, sizeof(float) * blurKernelWidth * blurKernelWidth);
  cudaMemcpy(d_filter, h_filter, sizeof(float) * blurKernelWidth * blurKernelWidth, cudaMemcpyHostToDevice);

  square_blur<<<gridSize, blockSize>>>(d_image, d_sq, d_filter, filterWidth, numRows, numCols, n_numRows, n_numCols);

  uchar4 *h_out = new uchar4[n_numRows * n_numCols * sizeof(uchar4)];
  cudaMemcpy(h_out, d_sq, n_numRows * n_numCols * sizeof(uchar4), cudaMemcpyDeviceToHost);
  return h_out; 
}
