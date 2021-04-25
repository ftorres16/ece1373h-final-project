#include "../src/cnn.h"
#include <iostream>

using namespace std;

int main() {
  float mem[512];

  int input_offset = 0;
  int output_offset = 10;
  int b = 1;
  int od = 2;
  int ox = 4;
  int oy = 4;
  int id = 3;
  int ix = 8;
  int iy = 8;
  int s = 1;
  int k = 1;

  cnn_layer(mem, input_offset, output_offset, b, od, ox, oy, id, ix, iy, s, k);

  cout << "Test successful.\n";
}
