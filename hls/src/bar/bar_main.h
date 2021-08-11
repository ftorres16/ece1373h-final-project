#ifndef BAR_MAIN_H
#define BAR_MAIN_H

#include "bar.h"

void bar_main(float *mem, const BAR_MEM_PARAMS mem_params, const int n_samples,
              const int *samples_offset, int *spikes_offset, int *n_spikes);

#endif
