#ifndef CONV_BATCH_RELU_MAX_H
#define CONV_BATCH_RELU_MAX_H

#include "conv.h"
#include "max_pool_2d.h"

void conv_batch_relu_max_layer(float *mem, const int params_offset,
                               const int input_offset, const int y0_offset,
                               const int output_offset,
                               CONV_LAYER_PARAMS conv_params,
                               MAX_POOL_2D_PARAMS max_pool_params);

void get_max_pool_stack_params(CONV_LAYER_PARAMS &conv_params,
                               MAX_POOL_2D_PARAMS *max_pool_params);
#endif
