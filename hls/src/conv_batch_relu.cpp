#include "batch_norm_2d.h"
#include "cnn.h"
#include "relu.h"

void conv_batch_relu_layer(float *mem, const int params_offset,
                           const int input_offset, const int output_offset,
                           const int b, const int od, const int ox,
                           const int oy, const int id, const int ix,
                           const int iy, const int s, const int kx,
                           const int ky, const int px, const int py) {

  cnn_layer(mem, params_offset, input_offset, output_offset, b, od, ox, oy, id,
            ix, iy, s, kx, ky, px, py);
  batch_norm_2d_layer(mem, output_offset, output_offset, b, od, ox, oy);
  relu_layer(mem, output_offset, output_offset, b * od * ox * oy);
}
