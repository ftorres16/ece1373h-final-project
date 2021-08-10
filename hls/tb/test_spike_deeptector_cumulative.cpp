#include "../src/spike_deeptector_cumulative.h"
#include <iostream>

using namespace std;

int main() {
  bool passed = true;

  int b = 7;
  int labels[] = {1, 0, 1, 0, 1, 0, 1};

  int out_label, out_label_gold = 1;
  float out_score, out_score_gold;

  out_score_gold = 4 / 7;

  spike_deeptector_cumulative(labels, &out_label, &out_score, b);

  if (out_label != out_label_gold) {
    passed = false;
    cout << "Error in the out label. Expected " << out_label_gold << " got "
         << out_label << endl;
  }

  if (out_score != out_score_gold) {
    passed = false;
    cout << "Error in the out score. Expected " << out_score_gold << " got "
         << out_score << endl;
  }

  if (passed) {
    cout << "SpikeDeeptector cumulative test successful. :)" << endl;
  } else {
    cout << "SpikeDeeptector cumulative test failed. :)" << endl;
  }
}
