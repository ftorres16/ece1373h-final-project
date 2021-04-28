#include <algorithm>

void fc_layer(float *mem, const int params_offset, const int input_offset,
              const int output_offset,
              const int b,  // batch size
              const int ox, // output width
              const int oy, // output height
              const int ix, // input width
              const int iy  // input height
) {

  int num_weights = ix * ox;

  // Batch
  for (int b_ = 0; b_ < b; b_++) {
    // Output Y dimension
    for (int o_y = 0; o_y < oy; o_y++) {
      // Output X dimension
      for (int o_x = 0; o_x < ox; o_x++) {
        // bias
        int b_addr = params_offset / sizeof(float) + num_weights + o_x;

        float output_element = mem[b_addr];

        for (int i_x = 0; i_x < ix; i_x++) {
          // output_element += x_ij * m_ij
          int x_ij_addr =
              input_offset / sizeof(float) + b_ * iy * ix + o_y * ix + i_x;
          int m_ij_addr = params_offset / sizeof(float) + i_x + o_x * oy;

          output_element += mem[x_ij_addr] * mem[m_ij_addr];
        }

        int out_addr =
            output_offset / sizeof(float) + b_ * oy * ox + o_y * ox + o_x;
        // mem[0] = std::max(0.0f, output_element);
        mem[out_addr] = output_element;
      }
    }
  }
}
