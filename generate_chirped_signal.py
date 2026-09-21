import os
import numpy as np
from scipy.signal import chirp

dir_path = os.path.dirname(os.path.realpath(__file__))
duration, fs = 100, 400  # 100 s signal with sampling frequency of 400 Hz
t = np.arange(int(fs*duration)) / fs  # timestamps of samples
signal = chirp(t, 100.0, t[-1], 20.0, method='logarithmic')
output_file = dir_path + "/long_decay_test_file.npy"
np.save(output_file,list(zip(t,signal)) )
print("Npy file with chirped wave was generated on file",output_file)
