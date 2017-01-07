#ifndef LOADSAVEIMAGE_H__
#define LOADSAVEIMAGE_H__

#include <string>
#include <cuda_runtime.h> // for uchar4

void loadImageRGBA(std::string &filename, uchar4 **imagePtr,
                   size_t *numRows, size_t *numCols);

void saveImageRGBA(uchar4* image, std::string &output_filename,
                   size_t numRows, size_t numCols);

unsigned char* saveImageGrey(unsigned char* image, std::string &output_filename,
                   size_t numRows, size_t numCols);

#endif