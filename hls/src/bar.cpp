#include "conv.h"
#include "conv_batch_relu.h"
#include "conv_batch_relu_max.h"
#include "conv_relu.h"
#include "zero_mean.h"

/*
 * Write the input into `mem_0`, then read it from `mem_1`.
 * Be careful about the sizes of memory.
 */

void bar(float *mem, const int params_offset, const int mem_0_offset,
         const int mem_1_offset, const int b, const int ix, const int iy) {

  int params_offset_0, params_offset_1, params_offset_2, params_offset_3,
      params_offset_fc_1, params_offset_fc_2, params_offset_fc_3;

  CONV_LAYER_PARAMS conv_stack_0, conv_stack_1, conv_stack_2, conv_fc_1,
      conv_fc_2, conv_fc_3;
  MAX_POOL_2D_PARAMS max_pool_stack_0, max_pool_stack_1;

  // Neural network layers config
  conv_stack_0.id = 1;
  conv_stack_0.od = 50;
  conv_stack_0.s = 1;
  conv_stack_0.kx = 3;
  conv_stack_0.ky = 1;
  conv_stack_0.px = 1;
  conv_stack_0.py = 0;
  conv_stack_0.b = b;
  conv_stack_0.ix = ix;
  conv_stack_0.iy = iy;
  get_conv_out_dims(&conv_stack_0);

  conv_stack_1.id = 50;
  conv_stack_1.od = 50;
  conv_stack_1.s = 1;
  conv_stack_1.kx = 5;
  conv_stack_1.ky = 1;
  conv_stack_1.px = 0;
  conv_stack_1.py = 0;
  conv_stack_1.b = conv_stack_0.b;
  conv_stack_1.ix = conv_stack_0.ox;
  conv_stack_1.iy = conv_stack_0.oy;
  get_conv_out_dims(&conv_stack_1);

  max_pool_stack_0.s = 1;
  max_pool_stack_0.kx = 2;
  max_pool_stack_0.ky = 1;
  get_max_pool_stack_params(conv_stack_1, &max_pool_stack_0);

  conv_stack_2.id = 50;
  conv_stack_2.od = 75;
  conv_stack_2.s = 1;
  conv_stack_2.kx = 5;
  conv_stack_2.ky = 1;
  conv_stack_2.px = 0;
  conv_stack_2.py = 0;
  conv_stack_2.b = max_pool_stack_0.b;
  conv_stack_2.ix = max_pool_stack_0.ox;
  conv_stack_2.iy = max_pool_stack_0.oy;
  get_conv_out_dims(&conv_stack_2);

  max_pool_stack_1.s = 1;
  max_pool_stack_1.kx = 2;
  max_pool_stack_1.ky = 1;
  get_max_pool_stack_params(conv_stack_2, &max_pool_stack_1);

  conv_fc_1.id = 75;
  conv_fc_1.od = 600;
  conv_fc_1.s = 1;
  conv_fc_1.kx = 38;
  conv_fc_1.ky = 1;
  conv_fc_1.px = 0;
  conv_fc_1.py = 0;
  conv_fc_1.b = max_pool_stack_1.b;
  conv_fc_1.ix = max_pool_stack_1.ox;
  conv_fc_1.iy = max_pool_stack_1.oy;
  get_conv_out_dims(&conv_fc_1);

  conv_fc_2.id = 600;
  conv_fc_2.od = 300;
  conv_fc_2.s = 1;
  conv_fc_2.kx = 1;
  conv_fc_2.ky = 1;
  conv_fc_2.px = 0;
  conv_fc_2.py = 0;
  conv_fc_2.b = conv_fc_1.b;
  conv_fc_2.ix = conv_fc_1.ox;
  conv_fc_2.iy = conv_fc_1.oy;
  get_conv_out_dims(&conv_fc_2);

  conv_fc_3.id = 300;
  conv_fc_3.od = 2;
  conv_fc_3.s = 1;
  conv_fc_3.kx = 1;
  conv_fc_3.ky = 1;
  conv_fc_3.px = 0;
  conv_fc_3.py = 0;
  conv_fc_3.b = conv_fc_2.b;
  conv_fc_3.ix = conv_fc_2.ox;
  conv_fc_3.iy = conv_fc_2.oy;
  get_conv_out_dims(&conv_fc_3);

  // Memory layout
  params_offset_0 = params_offset;
  params_offset_1 =
      params_offset_0 + conv_stack_0.ix * conv_stack_0.iy * sizeof(float);
  params_offset_2 =
      params_offset_1 +
      (get_conv_num_params(conv_stack_0) + 4 * conv_stack_0.od) * sizeof(float);
  params_offset_3 =
      params_offset_2 +
      (get_conv_num_params(conv_stack_1) + 4 * conv_stack_1.od) * sizeof(float);
  params_offset_fc_1 =
      params_offset_3 +
      (get_conv_num_params(conv_stack_2) + 4 * conv_stack_2.od) * sizeof(float);
  params_offset_fc_2 =
      params_offset_fc_1 + get_conv_num_params(conv_fc_1) * sizeof(float);
  params_offset_fc_3 =
      params_offset_fc_2 + get_conv_num_params(conv_fc_2) * sizeof(float);

  // Neural netowrk computation
  zero_mean_layer(mem, params_offset_0, mem_0_offset, mem_1_offset,
                  conv_stack_0.b, conv_stack_0.id, conv_stack_0.ix,
                  conv_stack_0.iy);

  conv_batch_relu_layer(mem, params_offset_1, mem_1_offset, mem_0_offset,
                        conv_stack_0);

  conv_batch_relu_max_layer(mem, params_offset_2, mem_0_offset, mem_1_offset,
                            conv_stack_1, max_pool_stack_0);

  conv_batch_relu_max_layer(mem, params_offset_3, mem_0_offset, mem_1_offset,
                            conv_stack_2, max_pool_stack_1);

  conv_relu_layer(mem, params_offset_fc_1, mem_0_offset, mem_1_offset,
                  conv_fc_1);

  conv_relu_layer(mem, params_offset_fc_2, mem_1_offset, mem_0_offset,
                  conv_fc_2);

  conv_layer(mem, params_offset_fc_3, mem_0_offset, mem_1_offset, conv_fc_3);
}
