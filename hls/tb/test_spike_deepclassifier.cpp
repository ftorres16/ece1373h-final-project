#include "../src/bar/bar.h"
#include "../src/spike_deepclassifier.h"
#include "../src/spike_deeptector/spike_deeptector.h"
#include "tb_config.h"
#include "utils.h"
#include <cmath>
#include <iostream>

int main() {
  bool passed = true;
  string src_file = CPP_ROOT_PATH "tb_data/spike_deepclassifier.txt";

  int mem_len = 2425252;
  int num_spike_deeptector_params = 449587;
  int num_bar_params = 1923825;
  int input_len = 2880;
  int mem_0_len = 24000;
  int mem_1_len = 24000;

  int output_len = mem_len - mem_0_len - mem_1_len - num_bar_params -
                   num_spike_deeptector_params;

  float *mem, *mem_gold;

  mem = (float *)malloc(mem_len * sizeof(float));
  mem_gold = (float *)malloc(mem_len * sizeof(float));

  if (!load_txt(mem_gold, src_file, mem_len)) {
    cout << "Could not load mem :(" << endl;
    return -1;
  }

  for (int i = 0; i < mem_len - output_len; i++) {
    mem[i] = mem_gold[i];
  }

  SPIKE_DEEPTECTOR_MEM_PARAMS sd_mem_params;
  BAR_MEM_PARAMS bar_mem_params;
  int inputs_offset;
  int outputs_offset;

  sd_mem_params.params_offset = 0;
  bar_mem_params.params_offset =
      sd_mem_params.params_offset + num_spike_deeptector_params * sizeof(float);
  inputs_offset = bar_mem_params.params_offset + num_bar_params * sizeof(float);

  sd_mem_params.mem_0_offset = inputs_offset + input_len * sizeof(float);
  sd_mem_params.mem_1_offset =
      sd_mem_params.mem_0_offset + mem_0_len * sizeof(float);
  bar_mem_params.mem_0_offset = sd_mem_params.mem_0_offset;
  bar_mem_params.mem_1_offset = sd_mem_params.mem_1_offset;

  outputs_offset = bar_mem_params.mem_0_offset + mem_1_len * sizeof(float);

  int n_electrodes = 2;
  int electrodes_offset[n_electrodes];

  electrodes_offset[0] = inputs_offset;
  electrodes_offset[1] = inputs_offset + int(2 * 48 * 20 * sizeof(float));
  electrodes_offset[2] = inputs_offset + int(3 * 48 * 20 * sizeof(float));

  spike_deepclassifier(mem, sd_mem_params, bar_mem_params, outputs_offset,
                       electrodes_offset, n_electrodes);

  int error_count = 0;
  bool flag = false;
  int first_failed_idx = 0;

  for (int i = 0; i < mem_len; i++) {
    // skip the check for mem buffers, because blocks are called in a different
    // order than the python tb_gen, only verify the output
    if (bar_mem_params.mem_0_offset <= i * sizeof(float) &&
        i < outputs_offset) {
      continue;
    }
    float diff = mem[i] - mem_gold[i];
    float avg = (mem[i] + mem_gold[i]) / 2;

    if (abs(diff) > .05 * abs(avg)) {
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

  cout << "===================================================================="
       << endl;
  cout << "inputs_offset: " << inputs_offset / sizeof(float) << endl;
  cout << "bar_mem_params_offset: "
       << bar_mem_params.params_offset / sizeof(float) << endl;
  cout << "spike_deeptector_mem_params_offset: "
       << sd_mem_params.params_offset / sizeof(float) << endl;
  cout << "mem_0_offset: " << bar_mem_params.mem_0_offset / sizeof(float)
       << endl;
  cout << "mem_1_offset: " << bar_mem_params.mem_1_offset / sizeof(float)
       << endl;
  cout << "output_offset: " << outputs_offset / sizeof(float) << endl;
  cout << "===================================================================="
       << endl;

  if (passed) {
    cout << "SpikeDeepClassifier test successful. :)" << endl;
    return 0;
  } else {
    cout << "SpikeDeepClassifier test failed. :(" << endl;
    cout << "First failed index: " << first_failed_idx << endl;
    cout << "Found " << error_count << " mismatching entries." << endl;
    return -1;
  }
}
