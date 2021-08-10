#include "../../src/layers/fc.h"
#include "../utils.h"
#include <iostream>
#include <map>
#include <string>

using namespace std;

int main() {
  bool passed = true;

  string src_file = "tb_data/fc.txt";
  string src_params = "tb_data/fc_params.txt";

  map<string, int> params = read_params(src_params);

  int b = params.at("b");
  int ix = params.at("ix");
  int iy = params.at("iy");
  int ox = params.at("ox");
  int oy = params.at("oy");

  // basic parameter validation
  if (b <= 0 || ix <= 0 || iy <= 0 || ox <= 0 || oy <= 0) {
    cout << "Invalid FC params :(" << endl;
    return -1;
  }

  int num_weights = ix * ox;
  int num_bias = ox;
  int num_inputs = b * iy * ix;
  int num_outputs = b * ox * oy;

  int params_offset = 0 * sizeof(float);
  int input_offset = params_offset + (num_weights + num_bias) * sizeof(float);
  int output_offset = input_offset + num_inputs * sizeof(float);

  int mem_len = num_weights + num_bias + num_inputs + num_outputs;

  float mem[mem_len];
  float mem_gold[mem_len];

  if (!load_txt(mem_gold, src_file, mem_len)) {
    cout << "Could not load mem :(" << endl;
    return -1;
  }

  for (int i = 0; i < mem_len - num_outputs; i++) {
    mem[i] = mem_gold[i];
  }

  fc_layer(mem, params_offset, input_offset, output_offset, b, ox, oy, ix, iy);

  for (int i = 0; i < mem_len; i++) {
    if (abs(mem[i] - mem_gold[i]) > abs(mem_gold[i]) * 0.01) {
      cout << "ERROR when comparing mem[" << i << "]. Expected: " << mem_gold[i]
           << " Got: " << mem[i] << endl;
      passed = false;
    }
  }

  if (passed) {
    cout << "Fully Connected test successful :)" << endl;
  } else {
    cout << "Fully Connected test failed :(" << endl;
    return -1;
  }
}
