#include "spike_deepclassifier.h"
#include "bar/bar.h"
#include "bar/bar_main.h"
#include "config.h"
#include "spike_deeptector/spike_deeptector.h"
#include "spike_deeptector/spike_deeptector_main.h"

#ifndef __SYNTHESIS__
#include <iostream>

using namespace std;
#endif

void spike_deepclassifier(
    float *mem, const SPIKE_DEEPTECTOR_MEM_PARAMS deeptector_mem_params,
    const BAR_MEM_PARAMS bar_mem_params, const int *electrodes_offset,
    const int n_electrodes) {
  // `pragmas` specified in directives.tcl so this layer can be used in
  // different projects

  int n_neural_channels = 0;
  int neural_channels[MAX_DEEPTECTOR_ELECTRODES];
  int samples_offset[MAX_DEEPTECTOR_SAMPLES];
  int spikes_offset[MAX_DEEPTECTOR_SAMPLES];

  spike_deeptector_main(mem, deeptector_mem_params, n_electrodes,
                        electrodes_offset, &n_neural_channels, neural_channels);

#ifndef __SYNTHESIS__
  // for testing purposes only
  cout << "Found " << n_neural_channels << " neural channels." << endl;
  for (int i = 0; i < n_neural_channels; i++) {
    cout << "Neural channel " << i << " at index " << neural_channels[i]
         << endl;
  }
#endif

  for (int i = 0; i < n_neural_channels; i++) {
    int n_samples;

    n_samples = get_n_samples(electrodes_offset, neural_channels[i]);

#ifndef __SYNTHESIS__
    cout << "Found " << n_samples << " samples for channel " << i << "."
         << endl;
#endif

    get_samples_offset(samples_offset, electrodes_offset, n_samples,
                       neural_channels[i]);

#ifndef __SYNTHESIS__
    // for debug purposes only
    for (int j = 0; j < n_samples; j++) {
      cout << "Sample " << j << " has offset " << samples_offset[j] << endl;
    }
#endif

    int n_spikes;

    // Processing for every neural channel.
    bar_main(mem, bar_mem_params, n_samples, samples_offset, &n_spikes,
             spikes_offset);

#ifndef __SYNTHESIS__
    // for testing purposes only
    cout << "Found " << n_spikes << " spikes." << endl;
    for (int j = 0; j < n_spikes; j++) {
      cout << "Spike number " << j << " at memory offset " << spikes_offset[j]
           << endl;
    }
#endif

    // pca and other code will come
  }
}

int get_n_samples(const int *electrodes_offset, const int idx) {
  return (electrodes_offset[idx + 1] - electrodes_offset[idx]) /
         (sizeof(float) * SAMPLE_LEN);
}

void get_samples_offset(int *samples_offset, const int *electrodes_offset,
                        const int n_samples, const int start_idx) {
  for (int i = 0; i < n_samples; i++) {
    samples_offset[i] =
        electrodes_offset[start_idx] + SAMPLE_LEN * i * sizeof(float);
  }
}
