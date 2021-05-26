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
               const int kx,            // kernel size x
               const int ky,            // kernel size y
               const int px,            // padding x
               const int py)            // padding y
{

  int num_weights = id * od * kx * ky;
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
            for (int i_y = o_y * s - py, iiy = 0; i_y < o_y * s - py + ky;
                 i_y++, iiy++) {
              // Input X Dimension
              for (int i_x = o_x * s - px, iix = 0; i_x < o_x * s - px + kx;
                   i_x++, iix++) {
                int k_i_addr = params_offset / sizeof(float) +
                               o_d * id * kx * ky + i_d * kx * ky + iiy * kx +
                               iix;
                int in_addr = input_offset / sizeof(float) + b_ * id * ix * iy +
                              i_d * ix * iy + i_y * ix + i_x;

                // accomodate for padding
                float in_val = (i_x >= 0 && i_x < ix) && (i_y >= 0 && i_y < iy)
                                   ? mem[in_addr]
                                   : 0;

                output_element += in_val * mem[k_i_addr];
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
