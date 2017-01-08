#ifndef FILTER_OPS_H__
#define FILTER_OPS_H__

uchar4* apply_filter(uchar4 *d_in, const size_t numRow, 
                     const size_t numCol, std::string filtername);

#endif