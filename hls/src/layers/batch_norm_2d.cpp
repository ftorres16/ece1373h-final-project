void batch_norm_2d_layer(float *mem, const int params_offset,
                         const int input_offset, const int output_offset,
                         const int b, const int id, const int ix,
                         const int iy) {
  float eps = 1e-6;
  float mu[id], std_dev[id], gamma[id], beta[id];
  int in_addr, out_addr;

  for (int i_d = 0; i_d < id; i_d++) {
    mu[i_d] = mem[params_offset / sizeof(float) + i_d];
    // Look out! Using standard dev instead of sigma to avoid sqrt()
    std_dev[i_d] = mem[params_offset / sizeof(float) + id + i_d];
    gamma[i_d] = mem[params_offset / sizeof(float) + 2 * id + i_d];
    beta[i_d] = mem[params_offset / sizeof(float) + 3 * id + i_d];
  }

  for (int i_d = 0; i_d < id; i_d++) {
    for (int b_ = 0; b_ < b; b_++) {
      for (int i_y = 0; i_y < iy; i_y++) {
        for (int i_x = 0; i_x < ix; i_x++) {
          in_addr = input_offset / sizeof(float) + b_ * id * iy * ix +
                    i_d * iy * ix + i_y * ix + i_x;
          out_addr = output_offset / sizeof(float) + b_ * id * iy * ix +
                     i_d * iy * ix + i_y * ix + i_x;

          mem[out_addr] =
              (mem[in_addr] - mu[i_d]) / (std_dev[i_d] + eps) * gamma[i_d] +
              beta[i_d];
        }
      }
    }
  }
}
