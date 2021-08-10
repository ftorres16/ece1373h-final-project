#ifndef SPIKE_DEEPTECTOR_SINGLE_RUN_H
#define SPIKE_DEEPTECTOR_SINGLE_RUN_H

#include "spike_deeptector.h"

void spike_deeptector_single_run(float *mem, const int params_offset,
                                 const int mem_0_offset, const int mem_1_offset,
                                 int *out,
                                 const SPIKE_DEPETECTOR_PARAMS params);

#endif
