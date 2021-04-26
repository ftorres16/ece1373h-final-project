#include "../src/cnn.h"
#include "utils.h"
#include <cmath>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string src_file = "tb_data/cnn.txt";
  bool passed = true;

  int b = 2;
  int id = 3;
  int ix = 2;
  int iy = 2;
  int od = 4;
  int s = 1;
  int k = 2;

  int ox = floor((ix - k) / s);
  int oy = floor((iy - k) / s);

  int num_weights = od * k * k * id;
  int num_bias = od;
  int num_inputs = b * id * ix * iy;
  int num_outputs = b * od * ox * oy;

  int input_offset = 0 * sizeof(float);
  int output_offset =
      input_offset + (num_weights + num_bias + num_inputs) * sizeof(float);

  int mem_len = num_weights + num_bias + num_inputs + num_outputs;
  float mem[mem_len];
  float mem_gold[mem_len];

  if (!load_txt(mem_gold, src_file)) {
    cout << "Could not load mem :(";
    return -1;
  }

  for (int i = 0; i < mem_len - num_outputs; i++) {
    mem[i] = mem_gold[i];
  }

  cnn_layer(mem, input_offset, output_offset, b, od, ox, oy, id, ix, iy, s, k);

  for (int i = 0; i < mem_len; i++) {
    if (abs(mem[i] - mem_gold[i]) > abs(mem_gold[i]) * 0.01) {
      cout << "ERROR when comparing mem[" << i << "]. Expected: " << mem_gold[i]
           << " Got: " << mem[i] << endl;
      passed = false;
    }
  }

  if (passed) {
    cout << "CNN test successful. :)" << endl;
  } else {
    cout << "CNN test failed :(" << endl;
    return -1;
  }
}
