#include "../src/full_nn.h"
#include "utils.h"
#include <iostream>
#include <map>

using namespace std;

int main() {
  bool passed = true;
  string src_file = "tb_data/full_nn.txt";

  int mem_len = 1959546;
  int num_inputs = 3400;
  int num_params = 448627;
  int num_intermediate_outputs = 1507225;
  int num_outputs = 294;

  float *mem, *mem_gold;

  mem = (float *)malloc(mem_len * sizeof(float));
  mem_gold = (float *)malloc(mem_len * sizeof(float));

  if (!load_txt(mem_gold, src_file, mem_len)) {
    cout << "Could not load mem :(" << endl;
    return -1;
  }

  for (int i = 0; i < mem_len - num_intermediate_outputs - num_outputs; i++) {
    mem[i] = mem_gold[i];
  }

  int input_offset = 0;
  int params_offset = input_offset + num_inputs * sizeof(float);
  int intermediate_results_offset = params_offset + num_params * sizeof(float);
  int output_offset =
      intermediate_results_offset + num_intermediate_outputs * sizeof(float);
  int b = 1;
  int ix = 50;
  int iy = 68;

  full_nn(mem, params_offset, input_offset, intermediate_results_offset,
          output_offset, b, ix, iy);

  int error_count = 0;
  bool flag = false;
  int first_failed_idx = 0;

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
    cout << "intermediate_results_offset: "
         << intermediate_results_offset / sizeof(float) << endl;
    return -1;
  }
}
