#include <fstream>
#include <iostream>
#include <map>
#include <string>

using namespace std;

bool load_txt(float *mem, const string fname, const int mem_len) {
  bool success = true;

  int i = 0;
  string line;
  ifstream my_file(fname, ios::in);

  if (my_file.is_open()) {
    while (getline(my_file, line)) {
      if (i < mem_len) {
        mem[i] = stof(line);
        i++;
      } else {
        cout << "Text file longer than expected! Was expecting: " << mem_len
             << " lines." << endl;
        success = false;
        break;
      }
    }
  } else {
    cout << "Unable to open file " << fname << endl;
    success = false;
  }

  my_file.close();

  return success;
}

map<string, int> read_params(const string fname) {
  map<string, int> params;
  string key, val;
  ifstream my_file(fname, ios::in);

  if (my_file.is_open()) {
    while (my_file >> key) {
      my_file >> val;
      params[key] = stoi(val);
    }
  } else {
    cout << "Unable to open file " << fname << endl;
  }

  my_file.close();

  return params;
}
