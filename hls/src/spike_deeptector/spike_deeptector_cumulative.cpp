void spike_deeptector_cumulative(const int *in_labels, int *out_label,
                                 float *out_score, const int b) {

  /*
   * Get a single label and confidence percentage for a prediction (single
   * electrode).
   */

  int num_0s = 0, num_1s = 0;

  for (int i = 0; i < b; i++) {
    if (in_labels[i] == 0)
      num_0s++;
    else
      num_1s++;
  }

  // Store the most common label and the percentage of samples it's active
  *out_label = num_0s > num_1s ? 0 : 1;
  *out_score = num_0s > num_1s ? num_0s / b : num_1s / b;
}
