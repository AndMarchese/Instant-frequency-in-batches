# Instant frequency in batches
This repository contains Python code that reads the instantaneous frequency of a large file by dividing it in batches, calculates their instantaneous frequency with the Hilbert transform, and recombines the results in a single file. This approach is for files that do not fit in the computer memory (where a single Hilbert transform is not feasible).

## Getting started

To use this code I recommend to create a new conda environment (for example with the name “hilbert_batches”)

`conda create --name hilbert_batches python`

where the required packages should be installed. After switching to the new environment (`conda activate hilbert_batches`), CuPy can be installed with (please be sure that the NVIDIA drivers and CUDA are installed correctly before this step):

`conda install -c conda-forge cupy`

and then the other dependencies can be installed with:

`pip install -r requirements.txt`

The file that contains the code to read the instantaneous frequency is “hilbert_istantaneous_frequency_in_batches.py”. The file “generate_chirped_signal.py“ can be used to create a test file with a monochromatic sine wave whose frequency decreases over time. The file “plot.py” can be used to display the result of the instantaneous frequency of all the files generated in the folder with the hilbert_istantaneous_frequency_in_batches.py” file, in comparison with the expected theoretical instantaneous frequency over time. Please notice that the theoretical instantaneous frequency is calculated in the “plot.py“ file following the parameters set in the “generate_chirped_signal.py“  file. Be careful to change the parameters for this calculation accordingly.
For a detailed explanation of how the modules work, see the comments in the code.

## Output examples

![alt text](https://github.com/AndMarchese/Instant-frequency-in-batches/blob/main/overlap_length.png)
As an example of the possible output of the code, figure “overlap_length.png” shows the calculated instantaneous frequency for different overlap lengths while keeping the batch size constant. We can see that for all the chosen overlap sizes, the curves describe well the expected decrease in frequency. For small overlap sizes (e.g. blue curve), the calculated frequency appears to be quite imprecise at the edges of the batches. This occurs because the reading of the frequency with the Hilbert transform does not perform well at the edges of the array, and a small overlap size does not effectively eliminate the most problematic part. On the other hand, at big overlap lengths (e.g. red curve) the error is reduced because with a larger overlap, we get rid of a larger portion of the edge of the array.
![alt text](https://github.com/AndMarchese/Instant-frequency-in-batches/blob/main/output_batch_length.png)
To give a further impression of the output, we can also see in figure “output_batch_length.png” the comparison of the output for different batch lenghts while keeping the overlap size constant. We can see that for small batches (e.g. blue curve), the Hilbert transform performs poorly, while it improves for larger batches (green curve).
For both plots, the output curves have different size. This occurs because the module rejects the batches that would have smaller size as being at the end of the file, and rejects the overlap regions at the beginning and at the end of the file.  
Ideally, the best reading of the instantaneous frequency for large files is obtained by setting the batch as large as permitted by the GPU memory, and the overlap as large as possible. However, in terms of performances a larger overlap means more batches to be calculated, which increases the calculation time.   
