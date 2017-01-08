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
const int BLOCK_WIDTH = 16;

// Kernel : Greyscale
__global__ void greyscale(const uchar4* const d_color, uchar4* d_grey, size_t numRows, size_t numCols)
{
  int thread_x = blockDim.x * blockIdx.x + threadIdx.x;
  int thread_y = blockDim.y * blockIdx.y + threadIdx.y;

  int myId = thread_y * numCols + thread_x;
  if(thread_x >= numCols || thread_y >= numRows) 
    return;
  
  unsigned char Y = 0.299f * d_color[myId].x + 0.587 * d_color[myId].y + 0.114 * d_color[myId].z;
  d_grey[myId] = make_uchar4(Y, Y, Y, 255);
}

__device__ double get_distance_between(double x1, double y1,
                                                double x2, double y2){
  return sqrt(pow(x1-x2, 2) + pow(y1-y2, 2));
}

// Kernel : Vignette
__global__ void vignette(const uchar4* const d_in, uchar4* d_vignette,
                         const size_t numRows, const size_t numCols, double max_dist)
{
  int thread_x = blockDim.x * blockIdx.x + threadIdx.x;
  int thread_y = blockDim.y * blockIdx.y + threadIdx.y;
  if(thread_x >= numCols || thread_y >= numRows)
    return;

  int myId = thread_y * numCols + thread_x; 
  
  // Generating mask
  const double max_image_radius = 1.0 * max_dist;
  const double power = 0.7;

  double dist_from_center = get_distance_between(numRows/2, numCols/2, thread_x, thread_y) / max_image_radius;  
  dist_from_center *= power;
  dist_from_center = pow(cos(dist_from_center), 4); 

  uchar4 rgba = d_in[myId];
  d_vignette[myId] = make_uchar4(rgba.x * dist_from_center, 
                                 rgba.y * dist_from_center,
                                 rgba.z * dist_from_center,
                                 255);
}

uchar4* apply_filter(uchar4 *d_in, const size_t numRows, const size_t numCols, std::string filtername)
{
  uchar4 *d_out;
  cudaMalloc((void **) &d_out, numRows * numCols * sizeof(uchar4));

  const dim3 block_size(BLOCK_WIDTH, BLOCK_WIDTH, 1);
  const dim3 grid_size(numCols/BLOCK_WIDTH + 1, numRows/BLOCK_WIDTH + 1, 1);

  if(filtername == "greyscale"){
    greyscale<<<grid_size, block_size>>>(d_in, d_out, numRows, numCols);
  }
  else if(filtername == "vignette"){
    double max_dist = sqrt((pow((double)numRows, 2) + pow((double)numCols, 2)) / 4);
    vignette<<<grid_size, block_size>>>(d_in, d_out, numRows, numCols, max_dist);
  }

  uchar4* h_out = new uchar4[numRows * numCols];
  cudaMemcpy(h_out, d_out, numRows * numCols * sizeof(uchar4), cudaMemcpyDeviceToHost);
  cudaFree(d_out);
  return h_out;
}