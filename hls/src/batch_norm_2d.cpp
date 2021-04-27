#include <cmath>

using namespace std;

void batch_norm_2d_layer(float *mem, const int input_offset,
                         const int output_offset, const int b, const int id,
                         const int ix, const int iy) {
  float eps = 1e-5;
  float mu[id], sigma[id];
  int in_addr, out_addr;

  for (int i_d = 0; i_d < id; i_d++) {
    mu[i_d] = 0;

    for (int b_ = 0; b_ < b; b_++) {
      for (int i_y = 0; i_y < iy; i_y++) {
        for (int i_x = 0; i_x < ix; i_x++) {
          in_addr = input_offset / sizeof(float) + b_ * id * iy * ix +
                    i_d * iy * ix + i_y * ix + i_x;

          mu[i_d] += mem[in_addr];
        }
      }
    }
    mu[i_d] /= b * iy * ix;
    sigma[i_d] = 0;

    for (int b_ = 0; b_ < b; b_++) {
      for (int i_y = 0; i_y < iy; i_y++) {
        for (int i_x = 0; i_x < ix; i_x++) {
          in_addr = input_offset / sizeof(float) + b_ * id * iy * ix +
                    i_d * iy * ix + i_y * ix + i_x;
          sigma[i_d] += (mem[in_addr] - mu[i_d]) * (mem[in_addr] - mu[i_d]);
        }
      }
    }

    sigma[i_d] /= b * iy * ix;
  }

  for (int i_d = 0; i_d < id; i_d++) {
    for (int b_ = 0; b_ < b; b_++) {
      for (int i_y = 0; i_y < iy; i_y++) {
        for (int i_x = 0; i_x < ix; i_x++) {
          in_addr = input_offset / sizeof(float) + b_ * id * iy * ix +
                    i_d * iy * ix + i_y * ix + i_x;
          out_addr = output_offset / sizeof(float) + b_ * id * iy * ix +
                     i_d * iy * ix + i_y * ix + i_x;

          mem[out_addr] = (mem[in_addr] - mu[i_d]) / sqrt(sigma[i_d] + eps);
        }
      }
    }
  }
}
