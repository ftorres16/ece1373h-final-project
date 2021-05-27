#include "../src/full_nn.h"
#include "utils.h"
#include <iostream>
#include <map>

using namespace std;

int main() {
  bool passed = true;
  string src_file = "tb_data/full_nn.txt";

  int mem_len = 6686371;
  int num_outputs = 0;

  // float mem[mem_len];
  // float mem_gold[mem_len];
  float *mem, *mem_gold;

  mem = (float *)malloc(mem_len * sizeof(float));
  mem_gold = (float *)malloc(mem_len * sizeof(float));

  if (!load_txt(mem_gold, src_file, mem_len)) {
    cout << "Could not load mem :(" << endl;
    return -1;
  }

  for (int i = 0; i < mem_len - num_outputs; i++) {
    mem[i] = mem_gold[i];
  }

  int params_offset = 0;
  int input_offset = 0;
  int intermediate_results_offset = 0;
  int output_offset = 0;
  int b = 1;
  int ix = 68;
  int iy = 50;

  full_nn(mem, params_offset, input_offset, intermediate_results_offset,
          output_offset, b, ix, iy);

  for (int i = 0; i < mem_len; i++) {
    if (abs(mem[i] - mem_gold[i]) > abs(mem_gold[i]) * 0.01) {
      cout << "ERROR when comparing mem[" << i << "]. Expected: " << mem_gold[i]
           << " Got: " << mem[i] << endl;
      passed = false;
    }
  }

  free(mem);
  free(mem_gold);

  if (passed) {
    cout << "Full NN test successful. :)" << endl;
  } else {
    cout << "Full NN test failed :(" << endl;
    return -1;
  }
}
