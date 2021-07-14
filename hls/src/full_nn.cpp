#include "conv.h"
#include "conv_batch_relu.h"
#include "conv_batch_relu_max.h"
#include "relu.h"

void full_nn(float *mem, const int params_offset, const int input_offset,
             const int intermediate_results_offset, const int output_offset,
             const int b, const int ix, const int iy) {

  int params_offset_0, params_offset_1, params_offset_2, params_offset_3,
      params_offset_4, params_offset_5, params_offset_fc_1, params_offset_fc_2;

  int output_offset_0 = 0, output_offset_1 = 0, y0_offset_1 = 0,
      output_offset_2 = 0, y0_offset_2 = 0, output_offset_3 = 0,
      y0_offset_3 = 0, output_offset_4 = 0, y0_offset_4 = 0,
      output_offset_5 = 0, y0_offset_5 = 0, output_offset_fc_1 = 0;

  CONV_LAYER_PARAMS conv_stack_0, conv_stack_1, conv_stack_2, conv_stack_3,
      conv_stack_4, conv_stack_5, conv_fc_1, conv_fc_2;
  MAX_POOL_2D_PARAMS max_pool_stack_1, max_pool_stack_2, max_pool_stack_3,
      max_pool_stack_4, max_pool_stack_5;

  // Neural network layers config
  conv_stack_0.id = 1;
  conv_stack_0.od = 25;
  conv_stack_0.s = 1;
  conv_stack_0.kx = 3;
  conv_stack_0.ky = 1;
  conv_stack_0.px = 1;
  conv_stack_0.py = 0;
  conv_stack_0.b = b;
  conv_stack_0.ix = ix;
  conv_stack_0.iy = iy;
  get_conv_out_dims(&conv_stack_0);

  conv_stack_1.id = 25;
  conv_stack_1.od = 25;
  conv_stack_1.s = 1;
  conv_stack_1.kx = 3;
  conv_stack_1.ky = 20;
  conv_stack_1.px = 0;
  conv_stack_1.py = 0;
  conv_stack_1.b = conv_stack_0.b;
  conv_stack_1.ix = conv_stack_0.ox;
  conv_stack_1.iy = conv_stack_0.oy;
  get_conv_out_dims(&conv_stack_1);

  max_pool_stack_1.s = 1;
  max_pool_stack_1.kx = 2;
  max_pool_stack_1.ky = 1;
  get_max_pool_stack_params(conv_stack_1, &max_pool_stack_1);

  conv_stack_2.id = 25;
  conv_stack_2.od = 50;
  conv_stack_2.s = 1;
  conv_stack_2.kx = 3;
  conv_stack_2.ky = 1;
  conv_stack_2.px = 0;
  conv_stack_2.py = 0;
  conv_stack_2.b = max_pool_stack_1.b;
  conv_stack_2.ix = max_pool_stack_1.ox;
  conv_stack_2.iy = max_pool_stack_1.oy;
  get_conv_out_dims(&conv_stack_2);

  max_pool_stack_2.s = 1;
  max_pool_stack_2.kx = 2;
  max_pool_stack_2.ky = 1;
  get_max_pool_stack_params(conv_stack_2, &max_pool_stack_2);

  conv_stack_3.id = 50;
  conv_stack_3.od = 100;
  conv_stack_3.s = 1;
  conv_stack_3.kx = 3;
  conv_stack_3.ky = 1;
  conv_stack_3.px = 0;
  conv_stack_3.py = 0;
  conv_stack_3.b = max_pool_stack_2.b;
  conv_stack_3.ix = max_pool_stack_2.ox;
  conv_stack_3.iy = max_pool_stack_2.oy;
  get_conv_out_dims(&conv_stack_3);

  max_pool_stack_3.s = 1;
  max_pool_stack_3.kx = 2;
  max_pool_stack_3.ky = 1;
  get_max_pool_stack_params(conv_stack_3, &max_pool_stack_3);

  conv_stack_4.id = 100;
  conv_stack_4.od = 100;
  conv_stack_4.s = 1;
  conv_stack_4.kx = 5;
  conv_stack_4.ky = 1;
  conv_stack_4.px = 0;
  conv_stack_4.py = 0;
  conv_stack_4.b = max_pool_stack_3.b;
  conv_stack_4.ix = max_pool_stack_3.ox;
  conv_stack_4.iy = max_pool_stack_3.oy;
  get_conv_out_dims(&conv_stack_4);

  max_pool_stack_4.s = 1;
  max_pool_stack_4.kx = 2;
  max_pool_stack_4.ky = 1;
  get_max_pool_stack_params(conv_stack_4, &max_pool_stack_4);

  conv_stack_5.id = 100;
  conv_stack_5.od = 100;
  conv_stack_5.s = 1;
  conv_stack_5.kx = 5;
  conv_stack_5.ky = 1;
  conv_stack_5.px = 0;
  conv_stack_5.py = 0;
  conv_stack_5.b = max_pool_stack_4.b;
  conv_stack_5.ix = max_pool_stack_4.ox;
  conv_stack_5.iy = max_pool_stack_4.oy;
  get_conv_out_dims(&conv_stack_5);

  max_pool_stack_5.s = 1;
  max_pool_stack_5.kx = 2;
  max_pool_stack_5.ky = 1;
  get_max_pool_stack_params(conv_stack_5, &max_pool_stack_5);

  conv_fc_1.id = 100;
  conv_fc_1.od = 100;
  conv_fc_1.s = 1;
  conv_fc_1.kx = 29;
  conv_fc_1.ky = 1;
  conv_fc_1.px = 0;
  conv_fc_1.py = 0;
  conv_fc_1.b = max_pool_stack_5.b;
  conv_fc_1.ix = max_pool_stack_5.ox;
  conv_fc_1.iy = max_pool_stack_5.oy;
  get_conv_out_dims(&conv_fc_1);

  conv_fc_2.id = 100;
  conv_fc_2.od = 2;
  conv_fc_2.s = 1;
  conv_fc_2.kx = 1;
  conv_fc_2.ky = 1;
  conv_fc_2.px = 0;
  conv_fc_2.py = 0;
  conv_fc_2.b = conv_fc_1.b;
  conv_fc_2.ix = conv_fc_1.ox;
  conv_fc_2.iy = conv_fc_1.oy;
  get_conv_out_dims(&conv_fc_2);

  // Memory layout
  params_offset_0 = params_offset;
  params_offset_1 =
      params_offset_0 +
      (get_conv_num_params(conv_stack_0) + 4 * conv_stack_0.od) * sizeof(float);
  params_offset_2 =
      params_offset_1 +
      (get_conv_num_params(conv_stack_1) + 4 * conv_stack_1.od) * sizeof(float);
  params_offset_3 =
      params_offset_2 +
      (get_conv_num_params(conv_stack_2) + 4 * conv_stack_2.od) * sizeof(float);
  params_offset_4 =
      params_offset_3 +
      (get_conv_num_params(conv_stack_3) + 4 * conv_stack_3.od) * sizeof(float);
  params_offset_5 =
      params_offset_4 +
      (get_conv_num_params(conv_stack_4) + 4 * conv_stack_4.od) * sizeof(float);
  params_offset_fc_1 =
      params_offset_5 +
      (get_conv_num_params(conv_stack_5) + 4 * conv_stack_5.od) * sizeof(float);
  params_offset_fc_2 =
      params_offset_fc_1 + get_conv_num_params(conv_fc_1) * sizeof(float);

  output_offset_0 = intermediate_results_offset;
  y0_offset_1 =
      output_offset_0 + get_conv_num_outputs(conv_stack_0) * sizeof(float);
  output_offset_1 =
      y0_offset_1 + get_conv_num_outputs(conv_stack_1) * sizeof(float);
  y0_offset_2 = output_offset_1 +
                get_max_pool_2d_num_outputs(max_pool_stack_1) * sizeof(float);
  output_offset_2 =
      y0_offset_2 + get_conv_num_outputs(conv_stack_2) * sizeof(float);
  y0_offset_3 = output_offset_2 +
                get_max_pool_2d_num_outputs(max_pool_stack_2) * sizeof(float);
  output_offset_3 =
      y0_offset_3 + get_conv_num_outputs(conv_stack_3) * sizeof(float);
  y0_offset_4 = output_offset_3 +
                get_max_pool_2d_num_outputs(max_pool_stack_3) * sizeof(float);
  output_offset_4 =
      y0_offset_4 + get_conv_num_outputs(conv_stack_4) * sizeof(float);
  y0_offset_5 = output_offset_4 +
                get_max_pool_2d_num_outputs(max_pool_stack_4) * sizeof(float);
  output_offset_5 =
      y0_offset_5 + get_conv_num_outputs(conv_stack_5) * sizeof(float);
  output_offset_fc_1 =
      output_offset_5 +
      get_max_pool_2d_num_outputs(max_pool_stack_5) * sizeof(float);

  // Neural network computation
  conv_batch_relu_layer(mem, params_offset_0, input_offset, output_offset_0,
                        conv_stack_0);

  conv_batch_relu_max_layer(mem, params_offset_1, output_offset_0, y0_offset_1,
                            output_offset_1, conv_stack_1, max_pool_stack_1);

  conv_batch_relu_max_layer(mem, params_offset_2, output_offset_1, y0_offset_2,
                            output_offset_2, conv_stack_2, max_pool_stack_2);

  conv_batch_relu_max_layer(mem, params_offset_3, output_offset_2, y0_offset_3,
                            output_offset_3, conv_stack_3, max_pool_stack_3);

  conv_batch_relu_max_layer(mem, params_offset_4, output_offset_3, y0_offset_4,
                            output_offset_4, conv_stack_4, max_pool_stack_4);

  conv_batch_relu_max_layer(mem, params_offset_5, output_offset_4, y0_offset_5,
                            output_offset_5, conv_stack_5, max_pool_stack_5);

  conv_layer(mem, params_offset_fc_1, output_offset_5, output_offset_fc_1,
             conv_fc_1);

  relu_layer(mem, output_offset_fc_1, output_offset_fc_1,
             conv_fc_1.b * conv_fc_1.od * conv_fc_1.ox * conv_fc_1.oy);

  conv_layer(mem, params_offset_fc_2, output_offset_fc_1, output_offset,
             conv_fc_2);
}
