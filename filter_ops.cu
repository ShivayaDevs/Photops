// Corresponding header file: /include/filter_ops.h
#include <cuda_runtime.h>
#include <string>

/* Write the code to apply instagram like filters to the image.
   Filter name/code will be received as a parameter.

   Decide the parameters for yourself and return a pointer to the new image.
   Or maybe, you can deallocate the memory of the incoming image after the operation.

   You will receive a pointer to h_inputImage so tasks like allocating memory 
   to GPU - you need to handle them yourself.  
*/
const int MAX_THREADS_PER_BLOCK = 512;

__global__ void greyscale(const uchar4* const d_color, unsigned char* d_grey, size_t num_pixels)
{
  int myId = threadIdx.x + blockIdx.x * blockDim.x;
  if(myId >= num_pixels) 
    return;
  d_grey[myId] = 0.299f * d_color[myId].x + 0.587 * d_color[myId].y + 0.114 * d_color[myId].z;
}

unsigned char* apply_filter(uchar4 **h_in, const size_t numRows, const size_t numCols, std::string filtername)
{
  // Allocate memory on GPU
  uchar4 *d_in;
  unsigned char *d_out;
  cudaMalloc((void **) &d_in, numRows * numCols * sizeof(uchar4));
  cudaMalloc((void **) &d_out, numRows * numCols * sizeof(unsigned char));

  // copy the input image onto the GPU
  cudaMemcpy(d_in, *h_in, numRows * numCols * sizeof(uchar4), cudaMemcpyHostToDevice);

  int threads = MAX_THREADS_PER_BLOCK;
  int blocks  = (numRows * numCols) / threads;

  if(filtername == "greyscale"){
    greyscale<<<blocks, threads>>>(d_in, d_out, numRows * numCols);
  }

  // copy the output back to the RAM
  unsigned char* h_out = new unsigned char[numRows * numCols];
  cudaMemcpy(h_out, d_out, numRows * numCols * sizeof(unsigned char), cudaMemcpyDeviceToHost);
  
  cudaFree(d_in);
  cudaFree(d_out);
  return h_out;
}