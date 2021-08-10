#ifndef SPIKE_DEEPTECTOR_H
#define SPIKE_DEEPTECTOR_H

typedef struct SPIKE_DEPETECTOR_PARAMS {
  int b;
  int ix;
  int iy;
} SPIKE_DEPETECTOR_PARAMS;

void spike_deeptector(float *mem, const int params_offset,
                      const int mem_0_offset, const int mem_1_offset,
                      const SPIKE_DEPETECTOR_PARAMS params);

#endif
