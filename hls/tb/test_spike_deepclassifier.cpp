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

  int mem_len = 2424292;
  int num_spike_deeptector_params = 449587;
  int num_bar_params = 1923825;
  int input_len = 2880;
  int mem_0_len = 24000;

  float *mem;

  mem = (float *)malloc(mem_len * sizeof(float));

  if (!load_txt(mem, src_file, mem_len)) {
    cout << "Could not load mem :(" << endl;
    return -1;
  }

  SPIKE_DEEPTECTOR_MEM_PARAMS sd_mem_params;
  BAR_MEM_PARAMS bar_mem_params;
  int inputs_offset;

  sd_mem_params.params_offset = 0;
  bar_mem_params.params_offset =
      sd_mem_params.params_offset + num_spike_deeptector_params * sizeof(float);
  inputs_offset = bar_mem_params.params_offset + num_bar_params * sizeof(float);

  sd_mem_params.mem_0_offset = inputs_offset + input_len * sizeof(float);
  sd_mem_params.mem_1_offset =
      sd_mem_params.mem_0_offset + mem_0_len * sizeof(float);
  bar_mem_params.mem_0_offset = sd_mem_params.mem_0_offset;
  bar_mem_params.mem_1_offset = sd_mem_params.mem_1_offset;

  int n_electrodes = 2;
  int electrodes_offset[n_electrodes];

  electrodes_offset[0] = inputs_offset;
  electrodes_offset[1] = inputs_offset + int(2 * 48 * 20 * sizeof(float));
  electrodes_offset[2] = inputs_offset + int(3 * 48 * 20 * sizeof(float));

  spike_deepclassifier(mem, sd_mem_params, bar_mem_params, electrodes_offset,
                       n_electrodes);

  free(mem);

  if (passed) {
    cout << "SpikeDeepClassifier test successful. :)" << endl;
    return 0;
  } else {
    cout << "SpikeDeepClassifier test failed. :(" << endl;
    return -1;
  }
}
