typedef struct CONV_LAYER_PARAMS {
  int b;  // batch size
  int od; // output dimensions
  int ox; // output width
  int oy; // output height
  int id; // input dimensions
  int ix; // input width
  int iy; // input height
  int s;  // stride
  int kx; // kernel size x
  int ky; // kernel size y
  int px; // padding x
  int py; // padding y
} CONV_LAYER_PARAMS;

void cnn_layer(float *mem,              // global memory pointer
               const int params_offset, // offset of parameters
               const int input_offset,  // offset of inputs
               const int output_offset, // offset of outputs
               CONV_LAYER_PARAMS params);
