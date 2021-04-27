#include "../src/relu.h"
#include "utils.h"
#include <cmath>
#include <iostream>
#include <map>
#include <string>

using namespace std;

int main() {
  bool success = true;

  string src_file = "tb_data/relu.txt";
  string src_params = "tb_data/relu_params.txt";

  map<string, int> params = read_params(src_params);

  int b = params.at("b");
  int ix = params.at("ix");
  int iy = params.at("iy");

  int num_inputs = b * iy * ix;
  int num_outputs = b * ix * iy;

  int input_offset = 0 * sizeof(float);
  int output_offset = input_offset + num_inputs * sizeof(float);

  int mem_len = num_inputs + num_outputs;

  float mem[mem_len];
  float mem_gold[mem_len];

  if (!load_txt(mem_gold, src_file, mem_len)) {
    cout << "Could not load mem :(";
    return -1;
  }

  for (int i = 0; i < mem_len - num_outputs; i++) {
    mem[i] = mem_gold[i];
  }

  relu_layer(mem, input_offset, output_offset, num_inputs);

  for (int i = 0; i < num_outputs; i++) {
    int addr = output_offset / sizeof(float) + i;

    if (mem[addr] < 0) {
      success = false;
      cout << addr << " is not >= 0\n";
    }
  }

  for (int i = 0; i < mem_len; i++) {
    if (abs(mem_gold[i] - mem[i]) > 0.1 * abs(mem[i])) {
      success = false;
      cout << "ERROR when comparing mem[" << i << "]. Expected: " << mem_gold[i]
           << " Got: " << mem[i] << endl;
    }
  }

  if (success) {
    cout << "ReLU test successful. :)" << endl;
  } else {
    cout << "ReLU test failed. :(" << endl;
  }
}
