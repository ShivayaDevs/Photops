#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <stdio.h>
#include <cuda_runtime.h>

using namespace std;
using namespace cv;

// This will return the image in the imagePtr variable
// It's the programmer's responsibility to free it after use.
void loadImageRGBA(string &filename, uchar4 **imagePtr,
                   size_t *numRows, size_t *numCols)
{
  // loading the image
  Mat image = imread(filename.c_str(), CV_LOAD_IMAGE_COLOR);
  if(image.empty()){
    cerr<<"Failed to load image: "<<filename<<endl;
    exit(1);
  }
  if(image.channels() != 3){
    cerr<<"Image must be color!"<<endl;
    exit(1);
  }
  if(!image.isContinuous()){
    cerr<<"Image isn't continuous!"<<endl;
    exit(1);
  }

  // forming a 4-channel(RGBA) image.
  Mat imageRGBA;
  cvtColor(image, imageRGBA, CV_BGR2RGBA);

  *imagePtr = new uchar4[image.rows * image.cols];
  unsigned char *cvPtr = imageRGBA.ptr<unsigned char>(0);
  for(size_t i = 0; i < image.rows * image.cols; ++i){
    (*imagePtr)[i].x = cvPtr[4*i + 0];
    (*imagePtr)[i].y = cvPtr[4*i + 1];
    (*imagePtr)[i].z = cvPtr[4*i + 2];
    (*imagePtr)[i].w = cvPtr[4*i + 3];
  }
  *numRows = image.rows;
  *numCols = image.cols;
}

void saveImageRGBA(uchar4* image, string &output_filename,
                   size_t numRows, size_t numCols)
{
  // Forming the Mat object from uchar4 array.
  int sizes[2] = {numRows, numCols};
  Mat imageRGBA(2, sizes, CV_8UC4, (void *)image);
  // Converting back to BGR system
  Mat imageOutputBGR;
  cvtColor(imageRGBA, imageOutputBGR, CV_RGBA2BGR);
  // Writing the image
  imwrite(output_filename.c_str(), imageOutputBGR);
}