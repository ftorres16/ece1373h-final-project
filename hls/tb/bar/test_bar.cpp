#include "../../src/bar/bar.h"
#include "../utils.h"
#include <iostream>
#include <map>

using namespace std;

int main() {
  bool passed = true;
  string src_file = "tb_data/bar.txt";
  string src_pre_file = "tb_data/bar_pre.txt";

  int mem_len = 1929600;
  int num_params = 1923825;
  int mem_0_len = 2850;

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
  int mem_0_offset = params_offset + num_params * sizeof(float);
  int mem_1_offset = mem_0_offset + mem_0_len * sizeof(float);
  int b = 1;
  int ix = 48;
  int iy = 1;

  bar(mem, params_offset, mem_0_offset, mem_1_offset, b, ix, iy);

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
    cout << "BAR test successful. :)" << endl;
  } else {
    cout << "BAR test failed :(" << endl;
    cout << "First failed index: " << first_failed_idx << endl;
    cout << "Found " << error_count << " mismatching entries." << endl;
    cout << "mem_0_offset: " << mem_0_offset / sizeof(float) << endl;
    cout << "mem_1_offset: " << mem_1_offset / sizeof(float) << endl;
    return -1;
  }
}
