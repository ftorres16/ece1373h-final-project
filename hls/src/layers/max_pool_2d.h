#ifndef MAX_POOL_2D_H
#define MAX_POOL_2D_H

typedef struct max_pool_2d_params {
  int b;  // batch size
  int od; // output dimension
  int id; // input dimension
  int ix; // input width
  int iy; // input height
  int s;  // stride
  int kx; // kernel size width
  int ky; // kernel size height
  // output dimensions, can be calculated with get_out_dims
  int ox; // output width
  int oy; // output height
} MAX_POOL_2D_PARAMS;

void max_pool_2d(float *mem, // global memory pointer
                 int input_offset, int output_offset,
                 MAX_POOL_2D_PARAMS params);

void get_max_pool_2d_out_dims(MAX_POOL_2D_PARAMS *params);

int get_max_pool_2d_num_inputs(MAX_POOL_2D_PARAMS params);
int get_max_pool_2d_num_outputs(MAX_POOL_2D_PARAMS params);

#endif
