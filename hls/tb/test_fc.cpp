#include "../src/fc.h"
#include <iostream>

using namespace std;

int main() {
  float mem[512];

  int input_offset = 0;
  int output_offset = 10;
  int b = 1;
  int ox = 4;
  int oy = 4;
  int ix = 8;
  int iy = 8;

  fc_layer(mem, input_offset, output_offset, b, ox, oy, ix, iy);

  cout << "Fully Connected test successful.\n";
}
