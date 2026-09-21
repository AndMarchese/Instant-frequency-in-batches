import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

filenames = []
rootdir = os.path.dirname(os.path.realpath(__file__)) + "/"

fig, (ax0) = plt.subplots(nrows=1, sharex='all')
ax0.set_title("Frequency of an amplitude-modulated chirp signal")
caption = "Output of the 'hilbert_istantaneous_frequency_in_batches' module \n" \
"for different output_batch_length at fixed overlap_length.\n" \
"The curves are vertically shifted for clarity."
ax0.set_ylabel("Frequency (Hz)")
ax0.set_yscale("log")
ax0.set_ylim([10,250])
ax0.set_xlabel("Time (s)")
# figure is extended to fit the caption below the image
fig.subplots_adjust(bottom=0.2)  
fig.text(.5, 0.015, caption, ha='center')

i = 1 #used to generate a waterfall plot
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        if file.startswith("long_decay_test_file_frequency") and file.endswith(".npy"):
            print( os.path.join(subdir,file),"added for plot")
            filenames.append(os.path.join(subdir,file))
            legend_string = str(Path(file).name).split("_frequency-")[-1].split(".npy")[0]
            data = np.load(rootdir + file)
            time = data[:,0]
            frequencies = data[:,1]
            vertical_shift = np.exp(i/5) # used for the waterfall plot
            ax0.plot(time,frequencies*vertical_shift, label=legend_string)
            i += 1
# calculation and plot of the frequency shift according to the law used in scipy.signal.chirp (used to generate the chirped signal)
# CAREFUL! check that this values matches with those used to generate the signal (in the "generate_chirped_signal.py")
f0 = 100 # Hz, initial frequency
f1 = 20 #Hz, final frequency
t1 = 100 #s, final time 
expected_frequencies = f0*(f1/f0)**(time/t1)
ax0.plot(time,expected_frequencies,'--', label='Generated frequency')

ax0.legend(loc=3)
plt.show()