// Corresponding header file: /include/filter_ops.h
#include <cuda_runtime.h>
#include <string>
#include <math.h>

/*
 * Contains kernels and functions for adding photo filters to the imput image.
 * apply_filter() function is called to apply the filter with image on GPU and
 * filter name as parameters. A pointer to the new image in RAM is returned to 
 * the caller. 
 */
const int MAX_THREADS_PER_BLOCK = 64;

// Kernel : Greyscale
__global__ void greyscale(const uchar4* const d_color, uchar4* d_grey, size_t num_pixels)
{
  int myId = threadIdx.x + blockIdx.x * blockDim.x;
  if(myId >= num_pixels) 
    return;
  
  unsigned char Y = 0.299f * d_color[myId].x + 0.587 * d_color[myId].y + 0.114 * d_color[myId].z;
  d_grey[myId] = make_uchar4(Y, Y, Y, 255);
}

__host__ __device__ double get_distance_between(double x1, double y1,
                                                double x2, double y2){
  return sqrt(pow(x1-x2, 2) + pow(y1-y2, 2));
}

// Kernel : Vignette
__global__ void vignette(const uchar4* const d_in, uchar4* d_vignette,
                         const size_t num_pixels, double max_dist)
{
  int myId = threadIdx.x + blockIdx.x * blockDim.x;
  if(myId >= num_pixels)
    return;

  // // Generating mask
  // const double max_image_radius = 1.0 * max_dist;
  // const double power = 0.8;

  // double dist_from_center = sqrt(pow((double)));//TODO: calculate
  // temp = (temp * power) / max_image_radius;
  // temp = pow(cos(temp), 4); 


  // uchar4 rgba = d_in[myId];
  // // Get RGBA to Lab
  // // Luminance *= temp 

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
  else if(filtername == "vignette"){
    double max_dist = sqrt((pow((double)numRows, 2) + pow((double)numCols, 2)) / 4);
    vignette<<<blocks, threads>>>(d_in, d_out, numRows * numCols, max_dist);
  }

  uchar4* h_out = new uchar4[numRows * numCols];
  cudaMemcpy(h_out, d_out, numRows * numCols * sizeof(uchar4), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  return h_out;
}