#include "bar.h"

void bar_main(float *mem, const BAR_MEM_PARAMS mem_params, const int n_samples,
              const int *samples_offset, int *spikes_offset, int *n_spikes) {

  BAR_PARAMS params;
  params.b = 1;
  params.ix = 48;
  params.iy = 1;

  *n_spikes = 0;

  // Iterate over all samples
  for (int s = 0; s < n_samples; s++) {

    for (int j = 0; j < params.b * params.ix * params.iy; j++) {
      // load inputs to `mem_0` for BAR
      int in_offset = samples_offset[s] / sizeof(float);
      int mem_0_offset = mem_params.mem_0_offset / sizeof(float);

      mem[mem_0_offset + j] = mem[in_offset + j];
    }

    bar(mem, mem_params, params);

    // Assign the labels preicted by the neural network
    // Fill a list with the memory addresses of all the spikes that are not
    // background activity
    int label_0_offset = mem_params.mem_1_offset / sizeof(float);
    int label_1_offset = mem_params.mem_1_offset / sizeof(float) + 1;

    if (mem[label_0_offset] > mem[label_1_offset]) {
      spikes_offset[*n_spikes] = samples_offset[s];
      (*n_spikes)++;
    }
  }
}
