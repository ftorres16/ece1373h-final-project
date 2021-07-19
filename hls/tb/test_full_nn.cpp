#include "../src/full_nn.h"
#include "utils.h"
#include <iostream>
#include <map>

using namespace std;

int main() {
  bool passed = true;
  string src_file = "tb_data/full_nn.txt";
  string src_pre_file = "tb_data/full_nn_pre.txt";

  // int mem_len = 855327;
  int mem_len = 855327;
  // int num_inputs = 3400;
  // int num_params = 448627;
  int num_params = 448627;
  // int num_outputs = 85000;

  // int mem_0_len = 205800;
  int mem_0_len = 205800;
  // int mem_1_len = 85000;

  float *mem, *mem_gold;

  mem = (float *)malloc(mem_len * sizeof(float));
  mem_gold = (float *)malloc(mem_len * sizeof(float));

  if (!load_txt(mem_gold, src_pre_file, mem_len)) {
    cout << "Could not load mem :(" << endl;
    return -1;
  }

  for (int i = 0; i < mem_len; i++) {
    mem[i] = mem_gold[i];
  }

  int params_offset = 0;
  int input_offset = params_offset + num_params * sizeof(float);
  int output_offset = input_offset + mem_0_len * sizeof(float);
  int b = 1;
  int ix = 50;
  int iy = 68;

  full_nn(mem, params_offset, input_offset, output_offset, b, ix, iy);

  int error_count = 0;
  bool flag = false;
  int first_failed_idx = 0;

  if (!load_txt(mem_gold, src_file, mem_len)) {
    cout << "Could not load mem :(" << endl;
    return -1;
  }

  for (int i = 0; i < mem_len; i++) {
    if (abs(mem[i] - mem_gold[i]) > 1e-4) {
      cout << "ERROR when comparing mem[" << i << "]. Expected: " << mem_gold[i]
           << " Got: " << mem[i] << endl;

      passed = false;
      error_count++;

      if (!flag) {
        flag = true;
        first_failed_idx = i;
      }
    }
  }

  free(mem);
  free(mem_gold);

  if (passed) {
    cout << "Full NN test successful. :)" << endl;
  } else {
    cout << "Full NN test failed :(" << endl;
    cout << "First failed index: " << first_failed_idx << endl;
    cout << "Found " << error_count << " mismatching entries." << endl;
    cout << "input offset: " << input_offset / sizeof(float) << endl;
    cout << "output offset: " << output_offset / sizeof(float) << endl;
    return -1;
  }
}
