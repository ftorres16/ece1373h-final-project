#include "../src/cnn.h"
#include "../src/relu.h"
#include "utils.h"
#include <cmath>
#include <iostream>
#include <map>
#include <string>

using namespace std;

int main() {
  bool passed = true;

  string src_file = "tb_data/conv_relu.txt";
  string src_params = "tb_data/conv_relu_params.txt";

  map<string, int> params = read_params(src_params);

  int b = params.at("b");
  int id = params.at("id");
  int ix = params.at("ix");
  int iy = params.at("iy");
  int od = params.at("od");
  int s = params.at("s");
  int k = params.at("k");
  int px = params.at("px");
  int py = params.at("py");

  // basic parameter validation
  if (b <= 0 || id <= 0 || ix <= 0 || iy <= 0 || od <= 0 || s <= 0 || k <= 0 ||
      px < 0 || py < 0) {
    cout << "Invalid Conv params :(" << endl;
    return -1;
  }

  int ox = floor((ix + 2 * px - k) / s + 1);
  int oy = floor((iy + 2 * py - k) / s + 1);

  int num_weights = od * k * k * id;
  int num_bias = od;
  int num_inputs = b * id * ix * iy;
  int num_outputs = b * od * ox * oy;

  int params_offset = 0 * sizeof(float);
  int input_offset = params_offset + (num_weights + num_bias) * sizeof(float);
  int output_offset_1 = input_offset + num_inputs * sizeof(float);
  int output_offset_2 = output_offset_1 + num_outputs * sizeof(float);

  int mem_len = num_weights + num_bias + num_inputs + num_outputs * 2;

  float mem[mem_len];
  float mem_gold[mem_len];

  if (!load_txt(mem_gold, src_file, mem_len)) {
    cout << "Could not load mem :(" << endl;
    return -1;
  }

  for (int i = 0; i < mem_len - num_outputs; i++) {
    mem[i] = mem_gold[i];
  }

  cnn_layer(mem, params_offset, input_offset, output_offset_1, b, od, ox, oy,
            id, ix, iy, s, k, k, px, py);
  relu_layer(mem, output_offset_1, output_offset_2, num_outputs);

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
