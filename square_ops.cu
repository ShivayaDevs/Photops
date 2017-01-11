// Corresponding header file: /include/square_ops.h
#include <cuda_runtime.h>
#include "include/blur_ops.h"
#include <stdio.h>

/* Image squaring operations.*/

//kernel to square an image
__global__ void kernel_square(uchar4 *d_in, uchar4 * d_out, size_t numRows, size_t numCols, uchar4 color){
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


//kernel to square blur an image
__global__
void kernel_blur(uchar4 *d_in, uchar4 *d_blur, uchar4 *d_out, size_t numRows, size_t numCols)
{
  int width = (numCols > numRows)? numCols: numRows;
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  //check out of bound
  if(x >= width || y >= width)
    return; 

  if(numCols > numRows)
  {
    int scaleFactor = numCols/numRows + 1;
    int shiftFactor = ( (numCols * scaleFactor) - width ) / 2 ;
    int w = (numCols - numRows) / 2;

    if(y >= w && y < width - w)
      d_out[y * width + x] = d_in[(y - w) * numCols + x];
    else
      d_out[y * width + x] = d_blur[y * numCols * scaleFactor + (x + shiftFactor)];

  }
  else
  {
    int scaleFactor = numRows/numCols + 1;
    int shiftFactor = ((numRows * scaleFactor) - width) / 2;
    int w = (numRows - numCols) / 2;

    if(x >= w && x < width - w)
      d_out[y * width + x] = d_in[y * numCols + (x - w)];
    else
      d_out[y * width + x] = d_blur[(y + shiftFactor) * numCols * scaleFactor + x];

  }
}

//kernel to zoom an image by scaling factor
__global__
void kernel_zoom(uchar4 * d_image, uchar4 * d_out, size_t numRows, size_t numCols, int scaleFactor)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if(x >= (numCols * scaleFactor) || y >= (numRows * scaleFactor))
    return ;

  //calculating nearest pixel
  int nearest_x = x / scaleFactor;
  int nearest_y = y / scaleFactor;

  d_out[y * numCols * scaleFactor + x] = d_image[nearest_y * numCols + nearest_x];
}

// function to square an image
uchar4* square_image(uchar4* const d_in, size_t &numRows, size_t &numCols, uchar4 color){

  size_t width = (numCols > numRows)? numCols : numRows;

  uchar4 *d_out;
  cudaMalloc((void **) &d_out, width * width * sizeof(uchar4));

  dim3 block_size(16, 16, 1);
  dim3 grid_size(width/block_size.x + 1, width/block_size.y + 1, 1);

  kernel_square<<<grid_size, block_size>>>(d_in, d_out, numRows, numCols, color);

  numRows = numCols = width;

  uchar4 *h_out = new uchar4[width * width * sizeof(uchar4)];
  cudaMemcpy(h_out, d_out, width * width * sizeof(uchar4), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  return h_out;   
}


// function to square blur an image
uchar4* square_blur(uchar4* d_image, size_t &numRows, size_t &numCols, int blurKernelWidth, float blurKernelSigma)
{
  dim3 threads(16, 16, 1);

  // calculating scaling factor
  int scaleFactor;
  if(numCols > numRows)
    scaleFactor = numCols/numRows + 1;
  else
    scaleFactor = numRows/numCols + 1;

  //new size: zoom matrice
  size_t newSize = numCols * numRows * scaleFactor * scaleFactor;
  
  dim3 zoom_grid(numCols * scaleFactor / threads.x + 1, numRows * scaleFactor / threads.y + 1, 1);

  //device zoom copy
  uchar4 *d_zoom;
  cudaMalloc(&d_zoom, sizeof(uchar4) * newSize);
  
  kernel_zoom<<<zoom_grid, threads>>>(d_image, d_zoom, numRows, numCols, scaleFactor);

  // blurring zoomed image
  uchar4 *h_blur = new uchar4[sizeof(uchar4) * newSize];
  h_blur = blur_ops(d_zoom, numRows * scaleFactor, numCols * scaleFactor, blurKernelWidth, blurKernelSigma);

  // device copy of zoom blur
  uchar4 * d_blur;
  cudaMalloc(&d_blur, sizeof(uchar4) * newSize);
  cudaMemcpy(d_blur, h_blur, sizeof(uchar4) * newSize, cudaMemcpyHostToDevice);

  size_t width = (numCols > numRows)? numCols: numRows;
  
  dim3 blocks(width/threads.x + 1, width/threads.y + 1, 1);

  uchar4 *d_out;
  cudaMalloc(&d_out, sizeof(uchar4) * width * width);

  kernel_blur<<<blocks, threads>>>(d_image, d_blur, d_out, numRows, numCols);

  numCols = numRows = width;

  uchar4 *h_out = new uchar4[width * width * sizeof(uchar4)];
  cudaMemcpy(h_out, d_out, sizeof(uchar4) * width * width, cudaMemcpyDeviceToHost);
  return h_out;
}
