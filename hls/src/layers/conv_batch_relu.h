#ifndef CONV_BATCH_RELU_H
#define CONV_BATCH_RELU_H

#include "conv.h"

void conv_batch_relu_layer(float *mem,              // global memory pointer
                           const int params_offset, // offset of parameters
                           const int input_offset,  // offset of inputs
                           const int output_offset, // offset of outputs
                           CONV_LAYER_PARAMS params // conv layer params
);

#define MAX_BATCH_NORM_CHANNELS 600
#endif
