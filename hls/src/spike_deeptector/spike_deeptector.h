#ifndef SPIKE_DEEPTECTOR_H
#define SPIKE_DEEPTECTOR_H

typedef struct SPIKE_DEPETECTOR_PARAMS {
  int b;
  int ix;
  int iy;
} SPIKE_DEPETECTOR_PARAMS;

typedef struct SPIKE_DEEPTECTOR_MEM_PARAMS {
  int params_offset;
  int mem_0_offset;
  int mem_1_offset;
} SPIKE_DEPETECTOR_MEM_PARAMS;

void spike_deeptector(float *mem, const SPIKE_DEEPTECTOR_MEM_PARAMS mem_params,
                      const SPIKE_DEPETECTOR_PARAMS params);

#endif
