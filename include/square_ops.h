#ifndef SQUARE_OPS_H__
#define SQUARE_OPS_H__

uchar4* square_image(uchar4* const d_in, size_t &numRows, size_t &numCols, uchar4 color);
uchar4* square_blur(uchar4* d_image, size_t &numRows, size_t &numCols, int blurKernelWidth, float blurKernelSigma);

#endif