#ifndef BAR_H
#define BAR_H

typedef struct BAR_MEM_PARAMS {
  int params_offset;
  int mem_0_offset;
  int mem_1_offset;
} SPIKE_DEPETECTOR_MEM_PARAMS;

typedef struct BAR_PARAMS {
  int b;
  int ix;
  int iy;
} BAR_PARAMS;

void bar(float *mem, const BAR_MEM_PARAMS mem_params, const BAR_PARAMS params);

#endif
