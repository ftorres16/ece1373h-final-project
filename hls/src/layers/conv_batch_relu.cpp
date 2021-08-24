#include "conv_batch_relu.h"
#include "conv.h"
#include <algorithm>

void conv_batch_relu_layer(float *mem, const int params_offset,
                           const int input_offset, const int output_offset,
                           const CONV_LAYER_PARAMS params) {
  // clang-format off

	// Global memory interface
	#pragma HLS INTERFACE m_axi port=mem depth=116 //update number
	// Bind all control ports to a single bundle
	#pragma HLS INTERFACE s_axilite port=params_offset bundle=CTRL_BUS
	#pragma HLS INTERFACE s_axilite port=input_offset bundle=CTRL_BUS
	#pragma HLS INTERFACE s_axilite port=output_offset bundle=CTRL_BUS
	#pragma HLS INTERFACE s_axilite port=return bundle=CTRL_BUS

  // clang-format on

  int num_conv_weights = get_conv_num_weights(params);
  int num_conv_bias = get_conv_num_bias(params);

  // Load batch norm params
  float mu[MAX_BATCH_NORM_CHANNELS], std_dev[MAX_BATCH_NORM_CHANNELS],
      gamma[MAX_BATCH_NORM_CHANNELS], beta[MAX_BATCH_NORM_CHANNELS];
  int bn_params_offset =
      params_offset / sizeof(float) + num_conv_weights + num_conv_bias;
  float eps = 1e-6;

  for (int o_d = 0; o_d < params.od; o_d++) {
    mu[o_d] = mem[bn_params_offset + o_d];
    // Look out! Using standard dev instead of sigma to avoid sqrt()
    std_dev[o_d] = mem[bn_params_offset + params.od + o_d];
    gamma[o_d] = mem[bn_params_offset + 2 * params.od + o_d];
    beta[o_d] = mem[bn_params_offset + 3 * params.od + o_d];
  }

  // Batch
  for (int b_ = 0; b_ < params.b; b_++) {
    // Output Dimensions
    for (int o_d = 0; o_d < params.od; o_d++) {
      // Output Y dimension
      for (int o_y = 0; o_y < params.oy; o_y++) {
        // Output X dimension
        for (int o_x = 0; o_x < params.ox; o_x++) {
          // Set bias
          float output_element =
              mem[params_offset / sizeof(float) + num_conv_weights + o_d];

          // Weighted Sum:

          // Input Dimensions (Feature Maps)
          for (int i_d = 0; i_d < params.id; i_d++) {
            // Input Y Dimension
            for (int i_y = o_y * params.s - params.py, iiy = 0;
                 i_y < o_y * params.s - params.py + params.ky; i_y++, iiy++) {
              // Input X Dimension
              for (int i_x = o_x * params.s - params.px, iix = 0;
                   i_x < o_x * params.s - params.px + params.kx; i_x++, iix++) {
                int k_i_addr = params_offset / sizeof(float) +
                               o_d * params.id * params.kx * params.ky +
                               i_d * params.kx * params.ky + iiy * params.kx +
                               iix;
                int in_addr = input_offset / sizeof(float) +
                              b_ * params.id * params.ix * params.iy +
                              i_d * params.ix * params.iy + i_y * params.ix +
                              i_x;

                // accomodate for padding
                float in_val = (i_x >= 0 && i_x < params.ix) &&
                                       (i_y >= 0 && i_y < params.iy)
                                   ? mem[in_addr]
                                   : 0;

                output_element += in_val * mem[k_i_addr];
              }
            }
          }

          // batch norm
          output_element =
              (output_element - mu[o_d]) / (std_dev[o_d] + eps) * gamma[o_d] +
              beta[o_d];

          // relu
          output_element = std::max(0.0f, output_element);

          // store in memory
          int out_addr = output_offset / sizeof(float) +
                         b_ * params.od * params.ox * params.oy +
                         o_d * params.ox * params.oy + o_y * params.ox + o_x;
          mem[out_addr] = output_element;
        }
      }
    }
  }
}
