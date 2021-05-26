#include "../src/conv_batch_relu_max.h"
#include "utils.h"
#include <cmath>
#include <iostream>
#include <map>

using namespace std;

int main() {
  bool passed = true;

  string src_file = "tb_data/conv_batch_relu_max.txt";
  string src_params = "tb_data/conv_batch_relu_max_params.txt";

  CONV_LAYER_PARAMS conv_params;
  MAX_POOL_2D_PARAMS max_pool_params;
  map<string, int> f_params = read_params(src_params);

  conv_params.b = f_params.at("b");
  conv_params.id = f_params.at("id");
  conv_params.ix = f_params.at("ix");
  conv_params.iy = f_params.at("iy");
  conv_params.od = f_params.at("od");
  conv_params.s = f_params.at("s");
  conv_params.kx = f_params.at("k");
  conv_params.ky = f_params.at("k");
  conv_params.px = f_params.at("px");
  conv_params.py = f_params.at("py");

  conv_params.ox = floor(
      (conv_params.ix + 2 * conv_params.px - conv_params.kx) / conv_params.s +
      1);
  conv_params.oy = floor(
      (conv_params.iy + 2 * conv_params.py - conv_params.ky) / conv_params.s +
      1);

  max_pool_params.s = f_params.at("max_pool_s");
  max_pool_params.kx = f_params.at("max_pool_kx");
  max_pool_params.ky = f_params.at("max_pool_ky");
  max_pool_params.b = conv_params.b;
  max_pool_params.id = conv_params.od;
  max_pool_params.ix = conv_params.ox;
  max_pool_params.iy = conv_params.oy;
  max_pool_params.od = conv_params.od;

  max_pool_params.ox =
      floor((conv_params.ox - max_pool_params.kx) / max_pool_params.s + 1);
  max_pool_params.oy =
      floor((conv_params.oy - max_pool_params.ky) / max_pool_params.s + 1);

  // basic parameter validation
  if (conv_params.b <= 0 || conv_params.id <= 0 || conv_params.ix <= 0 ||
      conv_params.iy <= 0 || conv_params.od <= 0 || conv_params.ox <= 0 ||
      conv_params.oy <= 0 || conv_params.s <= 0 || conv_params.kx <= 0 ||
      conv_params.ky <= 0 || conv_params.px < 0 || conv_params.py < 0) {
    cout << "Invalid Conv params :(" << endl;
    return -1;
  }

  if (max_pool_params.b <= 0 || max_pool_params.id <= 0 ||
      max_pool_params.ix <= 0 || max_pool_params.iy <= 0 ||
      max_pool_params.od <= 0 || max_pool_params.ox <= 0 ||
      max_pool_params.oy <= 0 || max_pool_params.s <= 0 ||
      max_pool_params.kx <= 0 || max_pool_params.ky <= 0) {
    cout << "Invalid Max Pool params :(" << endl;
    return -1;
  }

  int num_weights =
      conv_params.od * conv_params.kx * conv_params.ky * conv_params.id;
  int num_bias = conv_params.od;
  int num_inputs =
      conv_params.b * conv_params.id * conv_params.ix * conv_params.iy;
  int num_y0_outputs =
      conv_params.b * conv_params.od * conv_params.ox * conv_params.oy;
  int num_outputs = max_pool_params.b * max_pool_params.od *
                    max_pool_params.ox * max_pool_params.oy;

  int params_offset = 0 * sizeof(float);
  int input_offset = params_offset + (num_weights + num_bias) * sizeof(float);
  int y0_offset = input_offset + num_inputs * sizeof(float);
  int output_offset = y0_offset + num_y0_outputs * sizeof(float);

  int mem_len =
      num_weights + num_bias + num_inputs + num_y0_outputs + num_outputs;

  float mem[mem_len];
  float mem_gold[mem_len];

  if (!load_txt(mem_gold, src_file, mem_len)) {
    cout << "Could not load mem :(" << endl;
    return -1;
  }

  for (int i = 0; i < mem_len - num_y0_outputs - num_outputs; i++) {
    mem[i] = mem_gold[i];
  }

  conv_batch_relu_max_layer(mem, params_offset, input_offset, y0_offset,
                            output_offset, conv_params, max_pool_params);

  for (int i = 0; i < mem_len; i++) {
    if (abs(mem[i] - mem_gold[i]) > abs(mem_gold[i]) * 0.01) {
      cout << "ERROR when comparing mem[" << i << "]. Expected: " << mem_gold[i]
           << " Got: " << mem[i] << endl;
      passed = false;
    }
  }

  if (passed) {
    cout << "Conv + BatchNorm + ReLU + MaxPool test successful. :)" << endl;
  } else {
    cout << "Conv + BathNorm + ReLU + MaxPool test failed :(" << endl;
    return -1;
  }
}
