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

  MAX_POOL_2D_PARAMS params;
  map<string, int> f_params = read_params(src_params);

  params.b = f_params.at("b");
  params.id = f_params.at("id");
  params.ix = f_params.at("ix");
  params.iy = f_params.at("iy");
  params.s = f_params.at("s");
  params.kx = f_params.at("k");
  params.ky = f_params.at("k");

  // basic parameter validation
  if (params.b <= 0 || params.id <= 0 || params.ix <= 0 || params.iy <= 0 ||
      params.s <= 0 || params.kx <= 0 || params.ky <= 0) {
    cout << "Invalid MaxPool2D params :(" << endl;
    return -1;
  }

  params.od = params.id;
  params.ox = floor((params.ix - params.kx) / params.s + 1);
  params.oy = floor((params.iy - params.ky) / params.s + 1);

  int num_inputs = params.b * params.id * params.ix * params.iy;
  int num_outputs = params.b * params.od * params.ox * params.oy;

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

  max_pool_2d(mem, input_offset, output_offset, params);

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
