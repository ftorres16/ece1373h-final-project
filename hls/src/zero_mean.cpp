using namespace std;

void zero_mean_layer(float *mem, const int params_offset,
                     const int input_offset, const int output_offset,
                     const int b, const int id, const int ix, const int iy) {
  float mu;
  int in_addr, out_addr;

  for (int i_y = 0; i_y < iy; i_y++) {
    for (int i_x = 0; i_x < ix; i_x++) {

      // get one value of mean for each pixel position
      mu = mem[params_offset / sizeof(float) + ix * i_y + i_x];

      for (int i_d = 0; i_d < id; i_d++) {
        for (int b_ = 0; b_ < b; b_++) {
          in_addr = input_offset / sizeof(float) + b_ * id * iy * ix +
                    i_d * iy * ix + i_y * ix + i_x;
          out_addr = output_offset / sizeof(float) + b_ * id * iy * ix +
                     i_d * iy * ix + i_y * ix + i_x;

          mem[out_addr] = mem[in_addr] - mu;
        }
      }
    }
  }
}
