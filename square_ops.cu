// Corresponding header file: /include/square_ops.h
#include <cuda_runtime.h>
#include "include/blur_ops.h"
#include <stdio.h>

/* Image squaring operations.*/

__global__ void square_kernel(uchar4 *d_in, uchar4 * d_out, size_t numRows, size_t numCols, uchar4 color){
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
 
  int width = (numCols > numRows)? numCols:numRows;
  if(x >= width || y >= width)
    return;

  if(numCols>numRows){
    int w = (numCols - numRows) / 2 ;
    if(y >= w && y < width - w)
        d_out[y*numCols + x] = d_in[(y-w)*numCols + x];
    else
      d_out[y*numCols + x] = color;
  }
  else{
    int w = (numRows - numCols) / 2 ;
    if(x >= w && x < width - w)
      d_out[y*width + x] = d_in[y*numCols + x - w];
    else
      d_out[y*width + x] = color;
  }
}

__global__
void square_blur(const uchar4* d_in, uchar4* d_blur, uchar4* d_out, size_t numRows, size_t numCols)
{
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int width = (numRows > numCols)? numRows: numCols;

  if(x >= width || y >= width)  //check out of bounds
    return ;
  
  if(numCols > numRows)
  {
    int w = (numCols - numRows) / 2;
    if(y >= w && y < width - w)
      d_out[y * numCols + x] = d_in[(y-w) * numCols + x];
    else if(y < w)
      d_out[y * numCols + x] = d_blur[y * numCols + x];
    else
      d_out[y * numCols + x] = d_blur[(numRows + y - width) * numCols + x];

  }
  else
  {
    int w = (numRows - numCols) / 2;
    if(x >= w && x <= width - w )
      d_out[y * width + x] = d_in[y * numCols + (x - w)]; 
    else if(x < w)
      d_out[y * width + x] = d_blur[y * numCols + x];
    else
      d_out[y * width + x] = d_blur[y * numCols + (numCols + x - width)];
  }

}



uchar4* square_image(uchar4* const d_in, size_t &numRows, size_t &numCols, uchar4 color){

  size_t width = (numCols > numRows)? numCols : numRows;

  uchar4 *d_out;
  cudaMalloc((void **) &d_out, width * width * sizeof(uchar4));

  dim3 block_size(16, 16, 1);
  dim3 grid_size(width/block_size.x + 1, width/block_size.y + 1, 1);

  square_kernel<<<grid_size, block_size>>>(d_in, d_out, numRows, numCols, color);

  numRows = numCols = width;
  uchar4 *h_out = new uchar4[width * width * sizeof(uchar4)];
  cudaMemcpy(h_out, d_out, width * width * sizeof(uchar4), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  return h_out;   
}


uchar4* square_blur(uchar4* d_image, size_t &numRows, size_t &numCols, int blurKernelWidth, float blurKernelSigma)
{
	size_t width = (numRows > numCols)? numRows: numCols;
  const dim3 blockSize(16, 16, 1);  
  const dim3 gridSize(width/blockSize.x+1, width/blockSize.y+1, 1);  
  
  uchar4* h_blur = new uchar4[width * width * sizeof(uchar4)];

  //calling vagisha's function
  h_blur = blur_ops(d_in, numRows, numCols, 13, 3.0);
  //h_blur = blur_ops(d_in, numRows, numCols, 9, 2.0);

  uchar4* d_blur;
  cudaMalloc(&d_blur, sizeof(uchar4) * numRows * numCols);
  cudaMemcpy(d_blur, h_blur, sizeof(uchar4) * numRows * numCols, cudaMemcpyHostToDevice);

  uchar4* d_out;
  cudaMalloc(&d_out, sizeof(uchar4) * width * width);

  square_blur<<<gridSize, blockSize>>>(d_image, d_blur, d_out, numRows, numCols);

  uchar4 *h_out = new uchar4[width * width * sizeof(uchar4)];
  cudaMemcpy(h_out, d_out, width * width * sizeof(uchar4), cudaMemcpyDeviceToHost);
  return h_out; 
}
