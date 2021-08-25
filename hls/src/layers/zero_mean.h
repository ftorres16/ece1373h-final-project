#ifndef ZERO_MEAN_H
#define ZERO_MEAN_H

void zero_mean_layer(float *mem, const int params_offset,
                     const int input_offset, const int output_offset,
                     const int b, const int id, const int ix, const int iy);
#endif
