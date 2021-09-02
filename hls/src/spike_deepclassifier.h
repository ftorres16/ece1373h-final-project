#ifndef SPIKE_DEEPCLASSIFIER_H
#define SPIKE_DEEPCLASSIFIER_H

#include "bar/bar.h"
#include "spike_deeptector/spike_deeptector.h"

void spike_deepclassifier(
    float *mem, const SPIKE_DEEPTECTOR_MEM_PARAMS deeptector_mem_params,
    const BAR_MEM_PARAMS bar_mem_params, const int outputs_offset,
    const int *electrodes_offset, const int n_electrodes);

int get_n_samples(const int *electrodes_offset, const int idx);

void get_samples_offset(int *samples_offset, const int *electrodes_offset,
                        const int n_samples, const int start_idx);

#endif
