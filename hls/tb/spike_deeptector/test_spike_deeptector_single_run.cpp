#include "../../src/spike_deeptector/spike_deeptector.h"
#include "../../src/spike_deeptector/spike_deeptector_single_run.h"
#include "../utils.h"
#include <iostream>
#include <map>

using namespace std;

int main() {
  bool passed = true;
  string src_file = "tb_data/spike_deeptector.txt";
  string src_pre_file = "tb_data/spike_deeptector_pre.txt";

  int mem_len = 477587;
  int num_params = 449587;
  int mem_0_len = 24000;

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

  SPIKE_DEEPTECTOR_MEM_PARAMS mem_params;
  mem_params.params_offset = 0;
  mem_params.mem_0_offset =
      mem_params.params_offset + num_params * sizeof(float);
  mem_params.mem_1_offset = mem_params.mem_0_offset + mem_0_len * sizeof(float);

  SPIKE_DEEPETECTOR_PARAMS params;

  params.b = 1;
  params.ix = 48;
  params.iy = 20;

  int out[params.b];
  int out_gold[params.b] = {1};

  spike_deeptector_single_run(mem, mem_params, out, params);

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

  // check output
  for (int i = 0; i < params.b; i++) {
    if (out[i] != out_gold[i]) {
      cout << "Error when comparing out[" << i << "]. Expeted: " << out_gold[i]
           << " Got: " << out[i] << endl;
    }
  }

  if (passed) {
    cout << "SpikeDeeptector single run test successful. :)" << endl;
  } else {
    cout << "SpikeDeeptector single run test failed :(" << endl;
    cout << "First failed index: " << first_failed_idx << endl;
    cout << "Found " << error_count << " mismatching entries." << endl;
    cout << "mem_0 offset: " << mem_params.mem_0_offset / sizeof(float) << endl;
    cout << "mem_1 offset: " << mem_params.mem_1_offset / sizeof(float) << endl;
    return -1;
  }
}
