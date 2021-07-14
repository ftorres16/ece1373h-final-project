#include "conv.h"
#include "conv_batch_relu.h"
#include "max_pool_2d.h"
#include "relu.h"

void conv_batch_relu_max_layer(float *mem, const int params_offset,
                               const int input_offset, const int y0_offset,
                               const int output_offset,
                               CONV_LAYER_PARAMS conv_params,
                               MAX_POOL_2D_PARAMS max_pool_params) {

  conv_batch_relu_layer(mem, params_offset, input_offset, y0_offset,
                        conv_params);

  max_pool_2d(mem, y0_offset, output_offset, max_pool_params);
}

void get_max_pool_stack_params(CONV_LAYER_PARAMS &conv_params,
                               MAX_POOL_2D_PARAMS *max_pool_params) {
  /*
   * Fill all max pool 2d params that can be calculated from the convolutional
   * layer.
   */
  max_pool_params->id = conv_params.od;
  max_pool_params->b = conv_params.b;
  max_pool_params->ix = conv_params.ox;
  max_pool_params->iy = conv_params.oy;
  get_max_pool_2d_out_dims(max_pool_params);
}
