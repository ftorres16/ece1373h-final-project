#include "../src/conv_batch_relu.h"
#include "utils.h"
#include <cmath>
#include <iostream>
#include <map>

using namespace std;

int main() {
  bool passed = true;

  string src_file = "tb_data/conv_batch_relu.txt";
  string src_params = "tb_data/conv_batch_relu_params.txt";

  CONV_LAYER_PARAMS params;
  map<string, int> f_params = read_params(src_params);

  params.b = f_params.at("b");
  params.id = f_params.at("id");
  params.ix = f_params.at("ix");
  params.iy = f_params.at("iy");
  params.od = f_params.at("od");
  params.s = f_params.at("s");
  params.kx = f_params.at("kx");
  params.ky = f_params.at("ky");
  params.px = f_params.at("px");
  params.py = f_params.at("py");

  get_conv_out_dims(&params);

  // basic parameter validation
  if (params.b <= 0 || params.id <= 0 || params.ix <= 0 || params.iy <= 0 ||
      params.od <= 0 || params.ox <= 0 || params.oy <= 0 || params.s <= 0 ||
      params.kx <= 0 || params.ky <= 0 || params.px < 0 || params.py < 0) {
    cout << "Invalid Conv params :(" << endl;
    return -1;
  }

  int num_conv_weights = get_conv_num_weights(params);
  int num_conv_bias = get_conv_num_bias(params);
  int num_bn_weights = 4 * params.od;
  int num_inputs = get_conv_num_inputs(params);
  int num_outputs = get_conv_num_outputs(params);

  int params_offset = 0 * sizeof(float);
  int input_offset =
      params_offset +
      (num_conv_weights + num_conv_bias + num_bn_weights) * sizeof(float);
  int output_offset = input_offset + num_inputs * sizeof(float);

  int mem_len = num_conv_weights + num_conv_bias + num_bn_weights + num_inputs +
                num_outputs;

  float mem[mem_len];
  float mem_gold[mem_len];

  if (!load_txt(mem_gold, src_file, mem_len)) {
    cout << "Could not load mem :(" << endl;
    return -1;
  }

  for (int i = 0; i < mem_len - num_outputs; i++) {
    mem[i] = mem_gold[i];
  }

  conv_batch_relu_layer(mem, params_offset, input_offset, output_offset,
                        params);

  for (int i = 0; i < mem_len; i++) {
    if (abs(mem[i] - mem_gold[i]) > abs(mem_gold[i]) * 0.15) {
      cout << "ERROR when comparing mem[" << i << "]. Expected: " << mem_gold[i]
           << " Got: " << mem[i] << endl;
      passed = false;
    }
  }

  if (passed) {
    cout << "Conv + BatchNorm + ReLU test successful. :)" << endl;
  } else {
    cout << "Conv + BatchNorm + ReLU test failed :(" << endl;
    return -1;
  }
}
