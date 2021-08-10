#ifndef SPIKE_DEPETECTOR_MAIN_H
#define SPIKE_DEPETECTOR_MAIN_H

void spike_deeptector_main(float *mem, const int params_offset,
                           const int mem_0_offset, const int mem_1_offset,
                           const int in_offset, int *output_labels,
                           int *n_neural_channels, const int n_electrodes,
                           const int b);
#endif
