#include "../src/conv_batch_relu_max.h"
#include "../src/max_pool_2d.h"
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
  conv_params.kx = f_params.at("kx");
  conv_params.ky = f_params.at("ky");
  conv_params.px = f_params.at("px");
  conv_params.py = f_params.at("py");

  get_conv_out_dims(&conv_params);

  max_pool_params.s = f_params.at("max_pool_s");
  max_pool_params.kx = f_params.at("max_pool_kx");
  max_pool_params.ky = f_params.at("max_pool_ky");
  max_pool_params.b = conv_params.b;
  max_pool_params.id = conv_params.od;
  max_pool_params.ix = conv_params.ox;
  max_pool_params.iy = conv_params.oy;
  max_pool_params.od = conv_params.od;

  get_max_pool_2d_out_dims(&max_pool_params);

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

  int num_conv_weights = get_conv_num_weights(conv_params);
  int num_conv_bias = get_conv_num_bias(conv_params);
  int num_bn_weights = 4 * conv_params.od;
  int num_inputs = get_conv_num_inputs(conv_params);
  int num_y0_outputs = get_conv_num_outputs(conv_params);
  int num_outputs = get_max_pool_2d_num_outputs(max_pool_params);

  int params_offset = 0 * sizeof(float);
  int input_offset =
      params_offset +
      (num_conv_weights + num_conv_bias + num_bn_weights) * sizeof(float);
  int y0_offset = input_offset + num_inputs * sizeof(float);
  int output_offset = y0_offset + num_y0_outputs * sizeof(float);

  int mem_len = num_conv_weights + num_conv_bias + num_bn_weights + num_inputs +
                num_y0_outputs + num_outputs;

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
