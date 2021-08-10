#include "../../src/layers/conv_relu.h"
#include "../utils.h"
#include <cmath>
#include <iostream>
#include <map>
#include <string>

using namespace std;

int main() {
  bool passed = true;

  string src_file = "tb_data/conv_relu.txt";
  string src_params = "tb_data/conv_relu_params.txt";

  CONV_LAYER_PARAMS params;
  map<string, int> f_params = read_params(src_params);

  params.b = f_params.at("b");
  params.id = f_params.at("id");
  params.ix = f_params.at("ix");
  params.iy = f_params.at("iy");
  params.od = f_params.at("od");
  params.s = f_params.at("s");
  params.kx = f_params.at("k");
  params.ky = f_params.at("k");
  params.px = f_params.at("px");
  params.py = f_params.at("py");

  params.ox = floor((params.ix + 2 * params.px - params.kx) / params.s + 1);
  params.oy = floor((params.iy + 2 * params.py - params.ky) / params.s + 1);

  // basic parameter validation
  if (params.b <= 0 || params.id <= 0 || params.ix <= 0 || params.iy <= 0 ||
      params.od <= 0 || params.ox <= 0 || params.oy <= 0 || params.s <= 0 ||
      params.kx <= 0 || params.ky <= 0 || params.px < 0 || params.py < 0) {
    cout << "Invalid Conv params :(" << endl;
    return -1;
  }

  int num_weights = params.od * params.kx * params.ky * params.id;
  int num_bias = params.od;
  int num_inputs = params.b * params.id * params.ix * params.iy;
  int num_outputs = params.b * params.od * params.ox * params.oy;

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

  conv_relu_layer(mem, params_offset, input_offset, output_offset, params);

  for (int i = 0; i < mem_len; i++) {
    if (abs(mem[i] - mem_gold[i]) > abs(mem_gold[i]) * 0.01) {
      cout << "ERROR when comparing mem[" << i << "]. Expected: " << mem_gold[i]
           << " Got: " << mem[i] << endl;
      passed = false;
    }
  }

  if (passed) {
    cout << "Conv + ReLU test successful. :)" << endl;
  } else {
    cout << "Conv + ReLU test failed :(" << endl;
    return -1;
  }
}
