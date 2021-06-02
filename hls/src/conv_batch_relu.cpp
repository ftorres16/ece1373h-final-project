#include "batch_norm_2d.h"
#include "cnn.h"
#include "relu.h"

void conv_batch_relu_layer(float *mem, const int params_offset,
                           const int input_offset, const int output_offset,
                           CONV_LAYER_PARAMS params) {

  cnn_layer(mem, params_offset, input_offset, output_offset, params);
  batch_norm_2d_layer(mem, output_offset, output_offset, params.b, params.od,
                      params.ox, params.oy);
  relu_layer(mem, output_offset, output_offset,
             params.b * params.od * params.ox * params.oy);
}
