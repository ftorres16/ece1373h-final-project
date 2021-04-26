#include <algorithm>

void relu_layer(float *mem, int input_offset, int output_offset,
                const int num_entries) {
  for (int i = 0; i < num_entries; i++) {
    int in_addr = input_offset / sizeof(float) + i;
    int out_addr = output_offset / sizeof(float) + i;

    mem[out_addr] = std::max(0.0f, mem[in_addr]);
  }
}
