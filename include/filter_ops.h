#ifndef FILTER_OPS_H__
#define FILTER_OPS_H__

unsigned char* apply_filter(uchar4 **h_in, const size_t numRow, const size_t numCol, std::string filtername);

#endif