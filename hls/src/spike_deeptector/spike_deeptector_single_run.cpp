#include "spike_deeptector.h"

void spike_deeptector_single_run(float *mem,
                                 const SPIKE_DEEPTECTOR_MEM_PARAMS mem_params,
                                 int *out,
                                 const SPIKE_DEEPETECTOR_PARAMS params) {
  // `pragmas` specified in directives.tcl so this layer can be used in
  // different projects

  /*
   * params_offset, mem_0_offset, mem_1_offset are all for the deeptector
   * memory section. out_offset is where the results are stored.
   */

  spike_deeptector(mem, mem_params, params);

  for (int i = 0; i < params.b; i++) {
    // Assign the labels predicted by the neural network
    int label_0_offset = mem_params.mem_0_offset / sizeof(float) + 2 * i;
    int label_1_offset = mem_params.mem_0_offset / sizeof(float) + 2 * i + 1;

    out[i] = mem[label_0_offset] > mem[label_1_offset] ? 0 : 1;
  }
}
