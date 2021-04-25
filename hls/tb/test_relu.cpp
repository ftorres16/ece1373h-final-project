#include "../src/relu.h"
#include <cmath>
#include <iostream>

using namespace std;

bool run_single_test(int ix, int iy) {
  int input_offset = 0 * sizeof(float);
  int output_offset = ix * iy * sizeof(float);

  float mem[2 * ix * iy];

  bool success = true;

  relu_layer(mem, input_offset, output_offset, ix * iy);

  for (int i = 0; i < ix; i++) {
    for (int j = 0; j < iy; j++) {
      int addr = output_offset / sizeof(float) + i * iy + j;

      if (mem[addr] < 0) {
        success = false;

        cout << addr << " is not >= 0\n";
      }
    }
  }

  for (int i = 0; i < ix; i++) {
    for (int j = 0; j < iy; j++) {
      int in_addr = i * iy + j;
      int out_addr = output_offset / sizeof(float) + in_addr;

      float corrected_in = abs(mem[in_addr]);

      if (mem[in_addr] < 0)
        continue;

      if (abs(corrected_in - mem[out_addr]) > 0.1 * mem[out_addr]) {
        success = false;
        cout << in_addr << " does not match (" << mem[in_addr] << ", "
             << mem[out_addr] << ")\n";
      }
    }
  }

  return success;
}

int main() {
  bool success = true;

  for (int ix = 2; ix < 10; ix++) {
    for (int iy = 2; iy < 10; iy++) {
      if (!run_single_test(ix, iy)) {
        cout << "Error with (ix, iy): (" << ix << ", " << iy << ")\n";
        success = false;
      }
    }
  }

  if (success) {
    cout << "ReLU test successful. :)\n";
  } else {
    cout << "ReLU test failed. :(\n";
  }
}
