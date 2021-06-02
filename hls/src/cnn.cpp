#include "cnn.h"
#include <algorithm>
#include <cmath>

void cnn_layer(float *mem,               // global memory pointer
               const int params_offset,  // ofset of parameters
               const int input_offset,   // offset of inputs
               const int output_offset,  // offset of outputs
               CONV_LAYER_PARAMS params) // padding y
{

  int num_weights = params.id * params.od * params.kx * params.ky;
  // int num_input = b * id * ix * iy;

  // Batch
  for (int b_ = 0; b_ < params.b; b_++) {
    // Output Dimensions (Feature Maps)
    for (int o_d = 0; o_d < params.od; o_d++) {
      // Output Y Dimension
      for (int o_y = 0; o_y < params.oy; o_y++) {
        // Output X Dimension
        for (int o_x = 0; o_x < params.ox; o_x++) {
          // Set bias
          float output_element =
              mem[params_offset / sizeof(float) + num_weights + o_d];

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

          int out_addr = output_offset / sizeof(float) +
                         b_ * params.od * params.ox * params.oy +
                         o_d * params.ox * params.oy + o_y * params.ox + o_x;

          // Write output
          // mem[out_addr] = std::max(0.0f, output_element);
          mem[out_addr] = output_element;
        }
      }
    }
  }
}

void get_conv_out_dims(CONV_LAYER_PARAMS *params) {
  /*
   * Convenience function.
   * Calculate and load the output x and y dimensions into a params struct.
   */
  params->ox =
      floor((params->ix + 2 * params->px - params->kx) / params->s + 1);
  params->oy =
      floor((params->iy + 2 * params->py - params->ky) / params->s + 1);
}

int get_conv_num_weights(CONV_LAYER_PARAMS params) {
  return params.od * params.kx * params.ky * params.id;
}

int get_conv_num_bias(CONV_LAYER_PARAMS params) { return params.od; }

int get_conv_num_params(CONV_LAYER_PARAMS params) {
  return get_conv_num_weights(params) + get_conv_num_bias(params);
}

int get_conv_num_inputs(CONV_LAYER_PARAMS params) {
  return params.b * params.id * params.ix * params.iy;
}

int get_conv_num_outputs(CONV_LAYER_PARAMS params) {
  return params.b * params.od * params.ox * params.oy;
}
