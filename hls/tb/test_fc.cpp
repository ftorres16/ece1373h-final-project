#include "../src/fc.h"
#include <fstream>
#include <iostream>
#include <string>

using namespace std;

bool load_txt(float *mem) {
  int i = 0;
  string line;
  ifstream my_file("tb_data/fc.txt");

  if (my_file.is_open()) {
    while (getline(my_file, line)) {
      mem[i] = stof(line);
      i++;
    }
  } else {
    cout << "Unable to open file" << endl;
    return false;
  }

  return true;
}

int main() {
  int b = 2;
  int ix = 2;
  int iy = 2;
  int ox = 2;
  int oy = iy;

  int num_weights = ix * ox;
  int num_bias = ox;
  int num_inputs = b * iy * ix;
  int num_outputs = b * ox * oy;

  int input_offset = (num_weights + num_bias) * sizeof(float);
  int output_offset = input_offset + num_inputs * sizeof(float);

  int mem_len = num_weights + num_bias + num_inputs + num_outputs;
  float mem[mem_len];
  float mem_gold[mem_len];

  if (!load_txt(mem_gold)) {
    cout << "Could not load mem :(";
    return -1;
  }

  for (int i = 0; i < mem_len - num_outputs; i++) {
    mem[i] = mem_gold[i];
  }

  for (int i = 0; i < mem_len - num_outputs; i++) {
    if (abs(mem[i] - mem_gold[i]) > abs(mem_gold[i] * 0.1)) {
      cout << "ERROR when copying mem[" << i << "]. Expected: " << mem_gold[i]
           << " Got: " << mem[i] << endl;
      return -1;
    }
  }

  fc_layer(mem, input_offset, output_offset, b, ox, oy, ix, iy);

  bool passed = true;
  for (int i = 0; i < mem_len; i++) {
    if (abs(mem[i] - mem_gold[i]) > abs(mem_gold[i] * 0.1)) {
      cout << "ERROR when comparing mem[" << i << "]. Expected: " << mem_gold[i]
           << " Got: " << mem[i] << endl;
      passed = false;
    }
  }

  if (!passed) {
    return -1;
  }

  cout << "Fully Connected test successful.\n";
}
