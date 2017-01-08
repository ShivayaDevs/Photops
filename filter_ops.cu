// Corresponding header file: /include/filter_ops.h
#include <cuda_runtime.h>
#include <string>

/*
 * Contains kernels and functions for adding photo filters to the imput image.
 * apply_filter() function is called to apply the filter with image on GPU and
 * filter name as parameters. A pointer to the new image in RAM is returned to 
 * the caller. 
 */
const int MAX_THREADS_PER_BLOCK = 512;

__global__ void greyscale(const uchar4* const d_color, uchar4* d_grey, size_t num_pixels)
{
  int myId = threadIdx.x + blockIdx.x * blockDim.x;
  if(myId >= num_pixels) 
    return;
  
  unsigned char Y = 0.299f * d_color[myId].x + 0.587 * d_color[myId].y + 0.114 * d_color[myId].z;
  d_grey[myId] = make_uchar4(Y, Y, Y, 255);
}

uchar4* apply_filter(uchar4 *d_in, const size_t numRows, const size_t numCols, std::string filtername)
{
  uchar4 *d_out;
  cudaMalloc((void **) &d_out, numRows * numCols * sizeof(uchar4));

  int threads = MAX_THREADS_PER_BLOCK;
  int blocks  = (numRows * numCols) / threads + 1;

  if(filtername == "greyscale"){
    greyscale<<<blocks, threads>>>(d_in, d_out, numRows * numCols);
  }

  uchar4* h_out = new uchar4[numRows * numCols];
  cudaMemcpy(h_out, d_out, numRows * numCols * sizeof(uchar4), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  return h_out;
}