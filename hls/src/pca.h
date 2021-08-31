#ifndef PCA_H
#define PCA_H

#include "hls_linear_algebra.h"

// #define X_ROWS 2048
#define X_ROWS 512
#define X_COLS 48

void pca(float *mem, const int input_offset, const int output_offset,
         const int in_rows);

#endif
