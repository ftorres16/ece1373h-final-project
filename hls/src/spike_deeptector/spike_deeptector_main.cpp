#include "spike_deeptector.h"
#include "spike_deeptector_cumulative.h"
#include "spike_deeptector_single_run.h"

void spike_deeptector_main(float *mem,
                           const SPIKE_DEEPTECTOR_MEM_PARAMS mem_params,
                           const int in_offset, int *output_labels,
                           int *n_neural_channels, const int n_electrodes,
                           const int b) {

  float threshold = 0.85;

  int single_run_labels[b];
  int channel_labels[n_electrodes];
  float channel_scores[n_electrodes];

  SPIKE_DEPETECTOR_PARAMS params;
  params.b = b;
  params.ix = 48;
  params.iy = 20;

  *n_neural_channels = 0;

  // iterate over the electrodes
  for (int i = 0; i < n_electrodes; i++) {

    // load inputs for spike deeptector
    for (int j = 0; j < params.b * params.ix * params.iy; j++) {
      mem[mem_params.mem_0_offset / sizeof(float) + j] =
          mem[in_offset / sizeof(float) + i * params.b * params.ix * params.iy +
              j];
    }

    spike_deeptector_single_run(mem, mem_params, single_run_labels, params);

    spike_deeptector_cumulative(single_run_labels, &channel_labels[i],
                                &channel_scores[i], b);

    // Store only the spike neural channels
    if (channel_labels[i] == 0 && channel_scores[i] > threshold) {
      output_labels[*n_neural_channels] = i;
      (*n_neural_channels)++;
    }
  }
}
