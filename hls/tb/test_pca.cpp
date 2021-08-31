#include "../../hls/src/pca.h"
#include "tb_config.h"
#include "utils.h"
#include <cmath>
#include <iostream>
#include <map>
#include <string>

using namespace std;

int main() {
  bool passed = true;

  string src_file = CPP_ROOT_PATH "tb_data/pca.txt";
  string src_params = CPP_ROOT_PATH "tb_data/pca_params.txt";

  map<string, int> params = read_params(src_params);

  int in_rows = params.at("in_rows");

  // basic parameter validation
  if (in_rows <= 0) {
    cout << "Invalid PCA params :(" << endl;
    return -1;
  }

  int num_inputs = X_ROWS * X_COLS;
  int num_outputs = X_ROWS * X_COLS;

  int input_offset = 0;
  int output_offset = input_offset + num_inputs * sizeof(float);

  int mem_len = num_inputs + num_outputs;

  float mem[mem_len];
  float mem_gold[mem_len];

  if (!load_txt(mem_gold, src_file, mem_len)) {
    cout << "Could not load mem :(";
    return -1;
  }

  for (int i = 0; i < mem_len - num_outputs; i++) {
    mem[i] = mem_gold[i];
  }

  pca(mem, input_offset, output_offset, in_rows);

  int error_count = 0;
  bool flag = false;
  int first_failed_idx = 0;

  for (int i = 0; i < mem_len; i++) {
    float diff = mem[i] - mem_gold[i];
    float avg = (mem[i] + mem_gold[i]) / 2;

    if (abs(diff) > 0.5 * abs(avg)) {
      cout << "ERROR when comparing mem[" << i << "]. Expected: " << mem_gold[i]
           << " Got: " << mem[i] << endl;

      passed = false;
      error_count++;

      if (!flag) {
        flag = true;
        first_failed_idx = i;
      }
    }
  }

  if (passed) {
    cout << "PCA test successful. :)" << endl;
  } else {
    cout << "PCA test faild :(" << endl;
    cout << "First failed index: " << first_failed_idx << endl;
    cout << "Found " << error_count << " mismatching entries." << endl;
    cout << "input offset: " << input_offset / sizeof(float) << endl;
    cout << "output offset: " << output_offset / sizeof(float) << endl;
  }
}
