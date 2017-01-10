// Corresponding header file: /include/square_ops.h
#include <cuda_runtime.h>
#include "include/blur_ops.h"
#include <stdio.h>

/* Image squaring operations.*/

//kernel to square a image
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


//kernel to square blur a image
__global__
void kernel_blur(uchar4 *d_in, uchar4 *d_blur, uchar4 *d_out, size_t numRows, size_t numCols)
{
  int width = (numCols > numRows)? numCols: numRows;
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

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

}


//zooming image by scaling factor
void zoom(uchar4 *h_in, uchar4 *h_out, size_t numRows, size_t numCols, int scaleFactor)
{
  for(long cy = 0; cy < (numRows * scaleFactor); cy++)
  {
    for(long cx = 0; cx < (numCols * scaleFactor); cx++)
    {
      int pixel = (cy * numCols * scaleFactor) + cx;
      int y = cy/scaleFactor;
      int x = cx/scaleFactor;
      int nearest_pixel = (y * (numCols ) + x );
      
      h_out[pixel] = h_in[nearest_pixel];
    }
  }
}



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


uchar4* square_blur(uchar4* d_image, size_t &numRows, size_t &numCols, int blurKernelWidth, float blurKernelSigma)
{
  uchar4 *h_image = new uchar4[numCols * numRows * sizeof(uchar4)];
  cudaMemcpy(h_image, d_image, numRows * numCols * sizeof(uchar4), cudaMemcpyDeviceToHost);

  if(numCols > numRows)
  {
    int scaleFactor = numCols/numRows + 1;
    size_t newSize = numCols * numRows * scaleFactor * scaleFactor;

    uchar4 *h_zoom = new uchar4[sizeof(uchar4) * newSize];

    zoom(h_image, h_zoom, numRows, numCols, scaleFactor);


    uchar4 *d_zoom;
    cudaMalloc(&d_zoom, sizeof(uchar4) * newSize);
    cudaMemcpy(d_zoom, h_zoom, sizeof(uchar4) * newSize, cudaMemcpyHostToDevice);

    uchar4 *h_blur = new uchar4[sizeof(uchar4) * newSize];
    h_blur = blur_ops(d_zoom, numRows * scaleFactor, numCols * scaleFactor, blurKernelWidth, blurKernelSigma);

    uchar4 * d_blur;
    cudaMalloc(&d_blur, sizeof(uchar4) * newSize);
    cudaMemcpy(d_blur, h_blur, sizeof(uchar4) * newSize, cudaMemcpyHostToDevice);

    size_t width = (numCols > numRows)? numCols: numRows;
    dim3 threads(16, 16, 1);
    dim3 blocks(width/threads.x + 1, width/threads.y + 1, 1);

    uchar4 *d_out;
    cudaMalloc(&d_out, sizeof(uchar4) * width * width);
    kernel_blur<<<blocks, threads>>>(d_image, d_blur, d_out, numRows, numCols);

    numCols = numRows = width;

    uchar4 *h_out = new uchar4[width * width * sizeof(uchar4)];
    cudaMemcpy(h_out, d_out, sizeof(uchar4) * width * width, cudaMemcpyDeviceToHost);
    return h_out;

  }

}
