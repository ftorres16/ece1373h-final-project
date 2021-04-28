#include <algorithm>
#include <iostream>

void max_pool_2d(float *mem, // global memory pointer
                 const int input_offset, const int output_offset, const int b,
                 const int od, const int ox, const int oy, const int id,
                 const int ix, const int iy, const int s, const int k) {

  for (int b_ = 0; b_ < b; b_++) {
    for (int o_d = 0; o_d < id; o_d++) {
      for (int o_y = 0; o_y < oy; o_y++) {
        for (int o_x = 0; o_x < ox; o_x++) {

          float output_element;
          int addr;

          // input dimensions
          for (int i_y = o_y * s, iiy = 0; i_y < o_y * s + k; i_y++, iiy++) {
            for (int i_x = o_x * s, iix = 0; i_x < o_x * s + k; i_x++, iix++) {

              addr = input_offset / sizeof(float) + b_ * od * ix * iy +
                     o_d * ix * iy + i_y * ix + i_x;

              output_element = (iix == 0 && iiy == 0)
                                   ? mem[addr]
                                   : std::max(output_element, mem[addr]);
            }
          }

          addr = output_offset / sizeof(float) + b_ * od * ox * oy +
                 o_d * ox * oy + o_y * ox + o_x;
          mem[addr] = output_element;
        }
      }
    }
  }
}
