void spike_deeptector_cumulative(const int *in_labels, int *out_label,
                                 float *out_score, const int b) {

  /*
   * Get a single label and confidence percentage for a prediction (single
   * electrode).
   */

  int num_label_0 = 0, num_label_1 = 0;

  for (int i = 0; i < b; i++) {
    if (in_labels[i] == 0)
      num_label_0++;
    else
      num_label_1++;
  }

  // Store the most common label and the percentage of samples it's active
  *out_label = num_label_0 > num_label_1 ? 0 : 1;
  *out_score = num_label_0 > num_label_1 ? num_label_0 / b : num_label_1 / b;
}
