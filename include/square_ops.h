#ifndef SQUARE_OPS_H__
#define SQUARE_OPS_H__

// Declarations here
void square(const uchar4* d_in, uchar4* d_sq, size_t numRows, size_t numCols, size_t n_numRows, size_t n_numCols, uchar4 color);

void square_blur(const uchar4* d_in, uchar4* d_sq, const float* const d_filter, const int filterWidth, size_t numRows, size_t numCols, size_t n_numRows, size_t n_numCols);

uchar4* square(uchar4* const d_image, size_t numRows, size_t numCols, size_t &n_numRows, size_t &n_numCols, uchar4 color);

uchar4* square_blur(uchar4* const d_image, size_t numRows, size_t numCols, size_t &n_numRows, size_t &n_numCols, int blurKernelWidth, float blurKernelSigma);


#endif