#include "../../src/spike_deeptector/spike_deeptector.h"
#include "../../src/spike_deeptector/spike_deeptector_main.h"
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

  int n_electrodes = 1;
  int electrodes_offset[] = {
      mem_params.mem_0_offset,
      mem_params.mem_0_offset +
          int(48 * 20 * sizeof(float)) // adding input length
  };

  int n_neural_channels = 0;
  int output_labels[n_electrodes];
  int output_labels_gold[] = {};

  spike_deeptector_main(mem, mem_params, n_electrodes, electrodes_offset,
                        &n_neural_channels, output_labels);

  int error_count = 0;
  bool flag = false;
  int first_failed_idx = -1;

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

  // Test outputs
  if (n_neural_channels > n_electrodes) {
    passed = false;
    cout << "ERROR: More neural channels than electrodes." << endl;
  }

  for (int i = 0; i < n_neural_channels; i++) {
    if (output_labels[i] != output_labels_gold[i]) {
      passed = false;
      cout << "ERROR when comparing output_labels[" << i
           << "]. Expected: " << output_labels_gold[i]
           << " Got: " << output_labels[i] << endl;
    }
  }

  if (passed) {
    cout << "SpikeDeeptector main test successful. :)" << endl;
  } else {
    cout << "SpikeDeeptector main test failed :(" << endl;
    cout << "First failed index: " << first_failed_idx << endl;
    cout << "Found " << error_count << " mismatching entries." << endl;
    cout << "mem_0 offset: " << mem_params.mem_0_offset / sizeof(float) << endl;
    cout << "mem_1 offset: " << mem_params.mem_1_offset / sizeof(float) << endl;
    return -1;
  }
}
