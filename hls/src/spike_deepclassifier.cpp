#include "spike_deepclassifier.h"
#include "bar/bar.h"
#include "bar/bar_main.h"
#include "config.h"
#include "pca.h"
#include "spike_deeptector/spike_deeptector.h"
#include "spike_deeptector/spike_deeptector_main.h"

#ifndef __SYNTHESIS__
#include <iostream>

using namespace std;
#endif

void spike_deepclassifier(
    float *mem, const SPIKE_DEEPTECTOR_MEM_PARAMS deeptector_mem_params,
    const BAR_MEM_PARAMS bar_mem_params, const int outputs_offset,
    const int electrodes_addr_offset, const int n_electrodes) {
  // `pragmas` specified in directives.tcl so this layer can be used in
  // different projects

  int n_neural_channels = 0;

  int n_samples;
  int n_prev_samples = 0;
  int neural_channels[MAX_DEEPTECTOR_ELECTRODES];
  int samples_offset[MAX_DEEPTECTOR_SAMPLES];
  int spike_samples[MAX_DEEPTECTOR_SAMPLES];

  int channel_labels_offset;
  int bar_labels_offset, bar_labels_len;
  int pca_offset;

  // Memory addresses
  channel_labels_offset = outputs_offset;
  bar_labels_offset = channel_labels_offset + n_electrodes * sizeof(float);

  bar_labels_len = 0;
  for (int i = 0; i < n_electrodes; i++) {
    bar_labels_len += get_n_samples(mem, electrodes_addr_offset, i);
  }
  pca_offset = bar_labels_offset + bar_labels_len * sizeof(float);

  // Begin processing
  spike_deeptector_main(mem, deeptector_mem_params, n_electrodes,
                        electrodes_addr_offset, &n_neural_channels,
                        neural_channels);

  // Store deeptector outputs
  for (int i = 0; i < n_electrodes; i++) {
    float label = 1.0;

    for (int j = 0; j < n_neural_channels; j++) {
      if (neural_channels[j] == i) {
        label = 0.0;
      }
    }

    mem[channel_labels_offset / sizeof(float) + i] = label;
  }

#ifndef __SYNTHESIS__
  // for testing purposes only
  cout << "Found " << n_neural_channels << " neural channels." << endl;
  for (int i = 0; i < n_neural_channels; i++) {
    cout << "Neural channel " << i << " at index " << neural_channels[i]
         << endl;
  }
#endif

  for (int i = 0; i < n_neural_channels; i++) {
    n_samples = get_n_samples(mem, electrodes_addr_offset, neural_channels[i]);

#ifndef __SYNTHESIS__
    cout << "Found " << n_samples << " samples for neural channel " << i << "."
         << endl;
#endif

    get_samples_offset(mem, samples_offset, electrodes_addr_offset, n_samples,
                       neural_channels[i]);

    int n_spikes;

    // Processing for every neural channel.
    bar_main(mem, bar_mem_params, n_samples, samples_offset, &n_spikes,
             spike_samples);

    // Store BAR results
    for (int i_sample = 0; i_sample < n_samples; i_sample++) {
      float label = 1.0;

      for (int i_spike = 0; i_spike < n_spikes; i_spike++) {
        if (spike_samples[i_spike] == i_sample) {
          label = 0.0;
        }
      }
      mem[bar_labels_offset / sizeof(float) + n_prev_samples + i_sample] =
          label;
    }

#ifndef __SYNTHESIS__
    // for testing purposes only
    cout << "Found " << n_spikes << " spikes." << endl;
    for (int j = 0; j < n_spikes; j++) {
      cout << "Spike number " << j << " at memory offset " << spike_samples[j]
           << endl;
    }
#endif

    // PCA decomposition, to be used by kNN for finishign the classifier
    pca(mem, samples_offset[0],
        pca_offset + n_prev_samples * SAMPLE_LEN * sizeof(float), n_samples);

    n_prev_samples += n_samples;
  }
}

int get_n_samples(float *mem, const int electrodes_addr_offset, const int idx) {
  return (mem[electrodes_addr_offset / sizeof(float) + idx + 1] -
          mem[electrodes_addr_offset / sizeof(float) + idx]) /
         (sizeof(float) * SAMPLE_LEN);
}

void get_samples_offset(float *mem, int *samples_offset,
                        const int electrodes_addr_offset, const int n_samples,
                        const int start_idx) {
  for (int i = 0; i < n_samples; i++) {
    samples_offset[i] =
        mem[electrodes_addr_offset / sizeof(float) + start_idx] +
        SAMPLE_LEN * i * sizeof(float);
  }
}
