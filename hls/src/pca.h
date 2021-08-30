#include "hls_linear_algebra.h"

const int X_ROWS = 2048;
const int X_COLS = 48;

void pca(float *mem, const int input_offset, const int output_offset,
         const int in_rows, const int X_COLS);
