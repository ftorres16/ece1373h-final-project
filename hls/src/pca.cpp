#include "pca.h"

void pca(float *mem, const int input_offset, const int output_offset,
         const int in_rows) {

  float x_i[X_ROWS][X_COLS]; // Input X
  float x_mean_i[X_COLS];
  float cov_i[X_COLS][X_COLS];
  float s_i[X_ROWS][X_COLS];
  float u_i[X_COLS][X_COLS];
  float v_i[X_COLS][X_COLS];
  float pc_i[X_ROWS][X_COLS]; // principal components

  int in_addr = input_offset / sizeof(float);
  int out_addr = output_offset / sizeof(float);

// Copy input data from memory
x_row_loop:
  for (int r = 0; r < in_rows; r++) {
  x_col_loop:
    for (int c = 0; c < X_COLS; c++) {
      x_i[r][c] = mem[in_addr + r * X_COLS + c];
    }
  }

  // De-mean input data: Mean is calculated column-wise.
x_demean_init_loop:
  for (int i = 0; i < X_COLS; i++) {
    x_mean_i[i] = 0;
  }
x_demean_row_loop:
  for (int r = 0; r < in_rows; r++) {
  x_demean_col_loop:
    for (int c = 0; c < X_COLS; c++) {
      x_mean_i[c] += x_i[r][c] / X_ROWS;
    }
  }

//   Subtract mean from each row.
x_sub_row_loop:
  for (int r = 0; r < X_ROWS; r++) {
  x_sub_col_loop:
    for (int c = 0; c < X_COLS; c++) {
      x_i[r][c] = x_i[r][c] - x_mean_i[r]; // x_i is now de-mean-ed
    }
  }

  // Calculate covariance matrix
  hls::matrix_multiply<hls::Transpose, hls::NoTranspose, X_ROWS, X_COLS, X_ROWS,
                       X_COLS, X_COLS, X_COLS, float, float>(x_i, x_i, cov_i);

//   Divide by in_rows - 1
x_cov_row_loop:
  for (int r = 0; r < X_COLS; r++) {
  x_cov_col_loop:
    for (int c = 0; c < X_COLS; c++) {
      cov_i[r][c] = cov_i[r][c] / (in_rows - 1);
    }
  }

  // Call SVD
  hls::svd<X_COLS, X_COLS, float, float>(cov_i, s_i, u_i, v_i);

  // Get principal components From SVD
  //     : X = SUV'
  hls::matrix_multiply<hls::NoTranspose, hls::NoTranspose, X_ROWS, X_COLS,
                       X_COLS, X_COLS, X_ROWS, X_COLS, float, float>(x_i, v_i,
                                                                     pc_i);
// Copy output data to memory
pc_row_loop:
  for (int r = 0; r < in_rows; r++) {
  pc_col_loop:
    for (int c = 0; c < X_COLS; c++) {
      mem[out_addr + r * X_COLS + c] = pc_i[r][c];
    }
  }
}
