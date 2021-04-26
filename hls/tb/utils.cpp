#include <fstream>
#include <iostream>
#include <string>

using namespace std;

bool load_txt(float *mem, string fname) {
  int i = 0;
  string line;
  ifstream my_file(fname);

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
