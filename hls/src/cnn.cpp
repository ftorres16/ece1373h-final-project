#include "cnn.h"
#include <algorithm>

void cnn_layer(float *mem,              // global memory pointer
               const int params_offset, // ofset of parameters
               const int input_offset,  // offset of inputs
               const int output_offset, // offset of outputs
               const int b,             // batch size
               const int od,            // output dimensions
               const int ox,            // output width
               const int oy,            // output height
               const int id,            // input dimensions
               const int ix,            // input width
               const int iy,            // input height
               const int s,             // stride
               const int k)             // kernel size
{

  int num_weights = id * od * k * k;
  // int num_input = b * id * ix * iy;

  // Batch
  for (int b_ = 0; b_ < b; b_++) {
    // Output Dimensions (Feature Maps)
    for (int o_d = 0; o_d < od; o_d++) {
      // Output Y Dimension
      for (int o_y = 0; o_y < oy; o_y++) {
        // Output X Dimension
        for (int o_x = 0; o_x < ox; o_x++) {
          // Set bias
          float output_element =
              mem[params_offset / sizeof(float) + num_weights + o_d];

          // Weighted Sum:

          // Input Dimensions (Feature Maps)
          for (int i_d = 0; i_d < id; i_d++) {
            // Input Y Dimension
            for (int i_y = o_y * s, iiy = 0; i_y < o_y * s + k; i_y++, iiy++) {
              // Input X Dimension
              for (int i_x = o_x * s, iix = 0; i_x < o_x * s + k;
                   i_x++, iix++) {
                int k_i_addr = params_offset / sizeof(float) +
                               o_d * id * k * k + i_d * k * k + iiy * k + iix;
                int in_addr = input_offset / sizeof(float) + b_ * id * ix * iy +
                              i_d * ix * iy + i_y * ix + i_x;

                output_element += mem[in_addr] * mem[k_i_addr];
              }
            }
          }

          int out_addr = output_offset / sizeof(float) + b_ * od * ox * oy +
                         o_d * ox * oy + o_y * ox + o_x;

          // Write output
          // mem[out_addr] = std::max(0.0f, output_element);
          mem[out_addr] = output_element;
        }
      }
    }
  }
}
