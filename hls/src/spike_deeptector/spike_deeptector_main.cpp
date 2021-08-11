#include "spike_deeptector.h"
#include "spike_deeptector_cumulative.h"
#include "spike_deeptector_single_run.h"

void spike_deeptector_main(float *mem,
                           const SPIKE_DEEPTECTOR_MEM_PARAMS mem_params,
                           const int n_electrodes, const int *electrodes_offset,
                           int *n_neural_channels, int *output_labels) {

  /*
   * `electrodes_offset` is an array of length `n_electrodes + 1` with where the
   * signals for each element start. The last element is the end of the memory.
   */

  float threshold = 0.85;

  int channel_labels[n_electrodes];
  float channel_scores[n_electrodes];

  SPIKE_DEEPETECTOR_PARAMS params;
  params.b = 1;
  params.ix = 48;
  params.iy = 20;

  *n_neural_channels = 0;

  // iterate over the electrodes
  for (int i = 0; i < n_electrodes; i++) {

    int n_batches = (electrodes_offset[i + 1] - electrodes_offset[i]) /
                    (sizeof(float) * params.ix * params.iy);
    int single_run_labels[n_batches];

    for (int b_ = 0; b_ < n_batches; b_++) {

      // load inputs to `mem_0` for spike deeptector
      for (int j = 0; j < params.b * params.ix * params.iy; j++) {
        int in_offset =
            electrodes_offset[i] / sizeof(float) + b_ * params.ix * params.iy;
        int mem_0_offset = mem_params.mem_0_offset / sizeof(float);

        mem[mem_0_offset + j] = mem[in_offset + j];
      }

      spike_deeptector_single_run(mem, mem_params, &single_run_labels[b_],
                                  params);
    }

    spike_deeptector_cumulative(single_run_labels, &channel_labels[i],
                                &channel_scores[i], n_batches);

    // Store only the spike neural channels
    if (channel_labels[i] == 0 && channel_scores[i] > threshold) {
      output_labels[*n_neural_channels] = i;
      (*n_neural_channels)++;
    }
  }
}
