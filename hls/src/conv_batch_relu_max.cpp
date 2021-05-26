#include "batch_norm_2d.h"
#include "cnn.h"
#include "max_pool_2d.h"
#include "relu.h"

void conv_batch_relu_max_layer(float *mem, const int params_offset,
                               const int input_offset, const int y0_offset,
                               const int output_offset,
                               CONV_LAYER_PARAMS conv_params,
                               MAX_POOL_2D_PARAMS max_pool_params) {

  cnn_layer(mem, params_offset, input_offset, y0_offset, conv_params);
  batch_norm_2d_layer(mem, y0_offset, y0_offset, conv_params.b, conv_params.od,
                      conv_params.ox, conv_params.oy);
  relu_layer(mem, y0_offset, y0_offset,
             conv_params.b * conv_params.od * conv_params.ox * conv_params.oy);

  max_pool_2d(mem, y0_offset, output_offset, max_pool_params);
}
