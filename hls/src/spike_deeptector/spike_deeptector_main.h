#ifndef SPIKE_DEPETECTOR_MAIN_H
#define SPIKE_DEPETECTOR_MAIN_H

#include "spike_deeptector.h"

void spike_deeptector_main(float *mem,
                           const SPIKE_DEEPTECTOR_MEM_PARAMS mem_params,
                           const int n_electrodes,
                           const int electrodes_addr_offset[],
                           int *n_neural_channels, int *neural_channels);

#endif
