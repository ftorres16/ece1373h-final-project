#include <iostream>

using namespace std;

void zero_mean_layer(float *mem, const int params_offset,
                     const int input_offset, const int output_offset,
                     const int b, const int id, const int ix, const int iy) {
  // `pragmas` specified in directives.tcl so this layer can be used in
  // different projects

  float mu;
  int in_addr, out_addr, pixel_offset;

  for (int i_y = 0; i_y < iy; i_y++) {
    for (int i_x = 0; i_x < ix; i_x++) {

      // get one value of mean for each pixel position
      mu = mem[params_offset / sizeof(float) + i_y * ix + i_x];

      for (int i_d = 0; i_d < id; i_d++) {
        for (int b_ = 0; b_ < b; b_++) {
          pixel_offset = b_ * id * iy * ix + i_d * iy * ix + i_y * ix + i_x;
          in_addr = input_offset / sizeof(float) + pixel_offset;
          out_addr = output_offset / sizeof(float) + pixel_offset;

          mem[out_addr] = mem[in_addr] - mu;
        }
      }
    }
  }
}
