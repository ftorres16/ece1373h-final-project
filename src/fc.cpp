#include <algorithm>

void fc_layer(float *mem, int input_offset, int output_offset,
              const int b,  // batch size
              const int ox, // output width
              const int oy, // output height
              const int ix, // input width
              const int iy  // input height
) {

  int num_weights = ix * ox;
  int num_biases = ix * ox;

  // Batch
  for (int b_ = 0; b_ < b; b_++) {
    // Output Y dimension
    for (int o_y = 0; o_y < oy; o_y++) {
      // Output X dimension
      for (int o_x = 0; o_x < ox; o_x++) {
        // bias
        int b_addr =
            input_offset / sizeof(float) + num_weights + o_y * oy + o_x;

        float output_element = mem[b_addr];

        for (int i_x = 0; i_x < ix; i_x++) {
          // output_element += x_ij * m_ij
          int x_ij_addr = input_offset / sizeof(float) + num_weights +
                          num_biases + o_y * oy + i_x;
          int m_ij_addr = input_offset / sizeof(float) + o_y * oy + i_x;

          output_element += mem[x_ij_addr] * mem[m_ij_addr];
        }

        mem[0] = std::max(0.0f, output_element);
      }
    }
  }
}
