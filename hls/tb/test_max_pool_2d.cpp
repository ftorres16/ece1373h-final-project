#include "../src/max_pool_2d.h"
#include "utils.h"
#include <cmath>
#include <iostream>
#include <map>
#include <string>

using namespace std;

int main() {
  bool passed = true;

  string src_file = "tb_data/max_pool_2d.txt";
  string src_params = "tb_data/max_pool_2d_params.txt";

  map<string, int> params = read_params(src_params);

  int b = params.at("b");
  int id = params.at("id");
  int ix = params.at("ix");
  int iy = params.at("iy");
  int s = params.at("s");
  int k = params.at("k");

  // basic parameter validation
  if (b <= 0 || id <= 0 || ix <= 0 || iy <= 0 || s <= 0 || k <= 0) {
    cout << "Invalid MaxPool2D params :(" << endl;
    return -1;
  }

  int od = id;
  int ox = floor((ix - k) / s + 1);
  int oy = floor((iy - k) / s + 1);

  int num_inputs = b * id * ix * iy;
  int num_outputs = b * od * ox * oy;

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

  max_pool_2d(mem, input_offset, output_offset, b, od, ox, oy, id, ix, iy, s, k,
              k);

  for (int i = 0; i < mem_len; i++) {
    if (abs(mem[i] - mem_gold[i]) > abs(mem_gold[i]) * 0.01) {
      cout << "ERROR when comparing mem[" << i << "]. Expected: " << mem_gold[i]
           << " Got: " << mem[i] << endl;
      passed = false;
    }
  }

  if (passed) {
    cout << "Max Pool 2D test successful :)" << endl;
  } else {
    cout << "Max Pool 2D test failed :(" << endl;
  }
}
