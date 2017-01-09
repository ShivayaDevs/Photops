// Corresponding header file: /include/blur_ops.h
#include <cuda_runtime.h>
#include <stdio.h>     

unsigned char *d_red, *d_green, *d_blue;
float *d_filter;
uchar4 *d_inputImageRGBA__;
uchar4 *d_outputImageRGBA__;
float *h_filter__;

__global__
void gaussian_blur(const unsigned char* const inputChannel,
                   unsigned char* const outputChannel,
                   int numRows, int numCols, const float* const filter, const int filterWidth)
{
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  int row = blockIdx.y * blockDim.y + threadIdx.y;

  if ( col >= numCols || row >= numRows )
  {
   return;
  }

  float result = 0.f;
    //For every value in the filter around the pixel (c, r)
    for (int filter_r = -filterWidth/2; filter_r <= filterWidth/2; ++filter_r) 
    {
      for (int filter_c = -filterWidth/2; filter_c <= filterWidth/2; ++filter_c) 
      {
        //Find the global image position for this filter position
        //clamp to boundary of the image
        int image_r = min(max(row + filter_r, 0), static_cast<int>(numRows - 1));
        int image_c = min(max(col + filter_c, 0), static_cast<int>(numCols - 1));

        float image_value = static_cast<float>(inputChannel[image_r * numCols + image_c]);
        float filter_value = filter[(filter_r + filterWidth/2) * filterWidth + filter_c + filterWidth/2];

        result += image_value * filter_value;
      }
    }
  outputChannel[row * numCols + col] = result;
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




void allocateMemoryAndCopyToGPU(const size_t numRowsImage, const size_t numColsImage, const float* const h_filter, const size_t filterWidth)
{ //allocate memory for the three different channels
  //original
  cudaMalloc(&d_red,   sizeof(unsigned char) * numRowsImage * numColsImage);
  cudaMalloc(&d_green, sizeof(unsigned char) * numRowsImage * numColsImage);
  cudaMalloc(&d_blue,  sizeof(unsigned char) * numRowsImage * numColsImage);

  //Allocate memory for the filter on the GPU
  cudaMalloc(&d_filter, sizeof(float)*filterWidth*filterWidth);
  cudaMemcpy(d_filter,h_filter,sizeof(float)*filterWidth*filterWidth,cudaMemcpyHostToDevice);
}

void cleanup() {
  cudaFree(d_red);
  cudaFree(d_green);
  cudaFree(d_blue);
  cudaFree(d_filter);
}


void setFilter(float **h_filter, int *filterWidth, int blurKernelWidth, float blurKernelSigma)
{ //Normally blurKernelWidth = 9 and blurKernelSigma = 2.0 
  *h_filter = new float[blurKernelWidth * blurKernelWidth];
  *filterWidth = blurKernelWidth;
  float filterSum = 0.f; //for normalization

  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r)
   {
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) 
    {
      float filterValue = expf( -(float)(c * c + r * r) / (2.f * blurKernelSigma * blurKernelSigma));
      (*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] = filterValue;
      filterSum += filterValue;
    }
  }

  float normalizationFactor = 1.f / filterSum;
  for (int r = -blurKernelWidth/2; r <= blurKernelWidth/2; ++r) 
    for (int c = -blurKernelWidth/2; c <= blurKernelWidth/2; ++c) 
      (*h_filter)[(r + blurKernelWidth/2) * blurKernelWidth + c + blurKernelWidth/2] *= normalizationFactor;
}

uchar4* blur_ops(uchar4* d_inputImageRGBA, size_t numRows, size_t numCols, int blurKernelWidth, float blurKernelSigma)
{ //Set filter array
  float* h_filter;
  int filterWidth;
  setFilter(&h_filter, &filterWidth, blurKernelWidth, blurKernelSigma);

	//Set reasonable block size (i.e., number of threads per block)
  const dim3 blockSize(16,16,1);
  //Calculate Grid SIze
  int a=numCols/blockSize.x, b=numRows/blockSize.y;	
  const dim3 gridSize(a+1,b+1,1);
  const size_t numPixels = numRows * numCols;

  uchar4 *d_outputImageRGBA;
  cudaMalloc((void **)&d_outputImageRGBA, sizeof(uchar4) * numPixels);
  cudaMemset(d_outputImageRGBA, 0, numPixels * sizeof(uchar4)); //make sure no memory is left laying around

  d_inputImageRGBA__  = d_inputImageRGBA;
  d_outputImageRGBA__ = d_outputImageRGBA;

  //blurred
  unsigned char *d_redBlurred, *d_greenBlurred, *d_blueBlurred;
  cudaMalloc(&d_redBlurred,    sizeof(unsigned char) * numPixels);
  cudaMalloc(&d_greenBlurred,  sizeof(unsigned char) * numPixels);
  cudaMalloc(&d_blueBlurred,   sizeof(unsigned char) * numPixels);
  cudaMemset(d_redBlurred,   0, sizeof(unsigned char) * numPixels);
  cudaMemset(d_greenBlurred, 0, sizeof(unsigned char) * numPixels);
  cudaMemset(d_blueBlurred,  0, sizeof(unsigned char) * numPixels);

  allocateMemoryAndCopyToGPU(numRows, numCols, h_filter, filterWidth);

  //Launch a kernel for separating the RGBA image into different color channels
  separateChannels<<<gridSize, blockSize>>>(d_inputImageRGBA, numRows, numCols, d_red,d_green, d_blue);

  cudaDeviceSynchronize(); 

  //Call blur kernel here 3 times, once for each color channel.
  gaussian_blur<<<gridSize, blockSize>>>(d_red, d_redBlurred, numRows, numCols,  d_filter, filterWidth);
  gaussian_blur<<<gridSize, blockSize>>>(d_green, d_greenBlurred, numRows, numCols,  d_filter, filterWidth);
  gaussian_blur<<<gridSize, blockSize>>>(d_blue, d_blueBlurred, numRows, numCols,  d_filter, filterWidth);

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
  h_out = (uchar4*)malloc(sizeof(uchar4) * numPixels);

  //copy output from device to host
  cudaMemcpy(h_out, d_outputImageRGBA, sizeof(uchar4) * numPixels, cudaMemcpyDeviceToHost);

  cudaDeviceSynchronize(); 

  //cleanup memory on device
  cudaFree(d_inputImageRGBA__);
  cudaFree(d_outputImageRGBA__);
  delete[] h_filter__;

  //return h_out
	return h_out;
}