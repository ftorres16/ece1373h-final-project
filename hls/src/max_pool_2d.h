typedef struct max_pool_2d_params {
  int b;  // batch size
  int od; // output dimension
  int ox; // output width
  int oy; // output height
  int id; // input dimension
  int ix; // input width
  int iy; // input height
  int s;  // stride
  int kx; // kernel size width
  int ky; // kernel size height
} MAX_POOL_2D_PARAMS;

void max_pool_2d(float *mem, // global memory pointer
                 int input_offset, int output_offset,
                 MAX_POOL_2D_PARAMS params);
