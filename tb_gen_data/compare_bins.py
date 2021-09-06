import numpy as np

expected_file = "outputs/spike_deepclassifier.bin"
got_file = "SPIKEPRE.BIN"

MEM_OFFSET = 9697168 // 4

expected_data = np.fromfile(expected_file, np.float32)
got_data = np.fromfile(got_file, np.float32)

# truncate got data
assert np.all(got_data[len(expected_data) :] == 0.0)
got_data = got_data[: len(expected_data)]

diff = expected_data - got_data
avg = (expected_data + got_data) / 2

(mismatch_idxs,) = (np.abs(diff) > np.abs(avg) * 0.05).nonzero()

if mismatch_idxs[-1] > MEM_OFFSET:
    print("Mismatching indices: {mismatch_idxs[0]}")
else:
    print("Values match! :)")
