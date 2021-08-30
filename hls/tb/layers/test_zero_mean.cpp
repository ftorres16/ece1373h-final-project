#include "../../src/layers/zero_mean.h"
#include "../tb_config.h"
#include "../utils.h"
#include <cmath>
#include <iostream>
#include <map>
#include <string>

using namespace std;

int main() {
  bool success = true;

  string src_file = CPP_ROOT_PATH "tb_data/zero_mean.txt";
  string src_params = CPP_ROOT_PATH "tb_data/zero_mean_params.txt";

  map<string, int> params = read_params(src_params);

  int b = params.at("b");
  int id = params.at("id");
  int ix = params.at("ix");
  int iy = params.at("iy");

  // basic parameter validation
  if (b <= 0 || id <= 0 || ix <= 0 || iy <= 0) {
    cout << "Invalid batch norm params :(" << endl;
    return -1;
  }

  int num_inputs = b * id * iy * ix;
  int num_outputs = b * id * ix * iy;
  int num_params = ix * iy;

  int params_offset = 0 * sizeof(float);
  int input_offset = params_offset + num_params * sizeof(float);
  int output_offset = input_offset + num_inputs * sizeof(float);

  int mem_len = num_params + num_inputs + num_outputs;

  float mem[mem_len];
  float mem_gold[mem_len];

  if (!load_txt(mem_gold, src_file, mem_len)) {
    cout << "Could not load mem :(";
    return -1;
  }

  for (int i = 0; i < mem_len - num_outputs; i++) {
    mem[i] = mem_gold[i];
  }

  zero_mean_layer(mem, params_offset, input_offset, output_offset, b, id, ix,
                  iy);

  for (int i = 0; i < mem_len; i++) {
    if (abs(mem[i] - mem_gold[i]) > abs(mem_gold[i]) * 0.01) {
      success = false;
      cout << "Error when comparing mem[" << i << "]. Expected: " << mem_gold[i]
           << " Got: " << mem[i] << endl;
    }
  }

  if (success) {
    cout << "ZeroMean test successful :)" << endl;
  } else {
    cout << "ZeroMean test failed :(" << endl;
    return -1;
  }
}
