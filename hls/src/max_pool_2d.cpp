#include "max_pool_2d.h"
#include <algorithm>
#include <iostream>

void max_pool_2d(float *mem, // global memory pointer
                 const int input_offset, const int output_offset,
                 MAX_POOL_2D_PARAMS params) {

  for (int b_ = 0; b_ < params.b; b_++) {
    for (int o_d = 0; o_d < params.id; o_d++) {
      for (int o_y = 0; o_y < params.oy; o_y++) {
        for (int o_x = 0; o_x < params.ox; o_x++) {

          float output_element;
          int addr;

          // input dimensions
          for (int i_y = o_y * params.s, iiy = 0;
               i_y < o_y * params.s + params.ky; i_y++, iiy++) {
            for (int i_x = o_x * params.s, iix = 0;
                 i_x < o_x * params.s + params.kx; i_x++, iix++) {

              addr = input_offset / sizeof(float) +
                     b_ * params.od * params.ix * params.iy +
                     o_d * params.ix * params.iy + i_y * params.ix + i_x;

              output_element = (iix == 0 && iiy == 0)
                                   ? mem[addr]
                                   : std::max(output_element, mem[addr]);
            }
          }

          addr = output_offset / sizeof(float) +
                 b_ * params.od * params.ox * params.oy +
                 o_d * params.ox * params.oy + o_y * params.ox + o_x;
          mem[addr] = output_element;
        }
      }
    }
  }
}
