#ifndef CONV_H
#define CONV_H

typedef struct CONV_LAYER_PARAMS {
  int b;  // batch size
  int od; // output dimensions
  int id; // input dimensions
  int ix; // input width
  int iy; // input height
  int s;  // stride
  int kx; // kernel size x
  int ky; // kernel size y
  int px; // padding x
  int py; // padding y
  // output dimensions, can be calculated with get_out_dims
  int ox; // output width
  int oy; // output height
} CONV_LAYER_PARAMS;

void conv_layer(float *mem,              // global memory pointer
                const int params_offset, // offset of parameters
                const int input_offset,  // offset of inputs
                const int output_offset, // offset of outputs
                CONV_LAYER_PARAMS params);

void get_conv_out_dims(CONV_LAYER_PARAMS *params);

int get_conv_num_weights(CONV_LAYER_PARAMS params);
int get_conv_num_bias(CONV_LAYER_PARAMS params);
int get_conv_num_params(CONV_LAYER_PARAMS params);
int get_conv_num_inputs(CONV_LAYER_PARAMS params);
int get_conv_num_outputs(CONV_LAYER_PARAMS params);

#endif
