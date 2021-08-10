#include "spike_deeptector.h"

void spike_deeptector_single_run(float *mem, const int params_offset,
                                 const int mem_0_offset, const int mem_1_offset,
                                 int *out,
                                 const SPIKE_DEPETECTOR_PARAMS params) {

  /*
   * params_offset, mem_0_offset, mem_1_offset are all for the deeptector
   * memory section. out_offset is where the results are stored.
   */

  spike_deeptector(mem, params_offset, mem_0_offset, mem_1_offset, params);

  for (int i = 0; i < params.b; i++) {
    // Assign the labels predicted by the neural network
    int label_0_offset = mem_0_offset / sizeof(float) + 2 * i;
    int label_1_offset = mem_0_offset / sizeof(float) + 2 * i + 1;

    out[i] = mem[label_0_offset] > mem[label_1_offset] ? 0 : 1;
  }
}