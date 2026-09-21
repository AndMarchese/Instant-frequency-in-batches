import os
from pathlib import PurePath

import cupy as cp
import numpy as np
from cupyx.scipy.signal import hilbert as cupy_hilbert
from npy_append_array import (
    NpyAppendArray,  # this library is used to write sequentially pieces of ndarray on a npy file
)
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm


def hilbert_istantaneous_frequency_in_batches(filename : str, output_length_batch : int, overlap_length : int):
    """
    Hilbert transform istantaneous frequency reading for large files.
    This module reads amplitude vs time data from a numpy file in 'filename' of a chirped signal, 
    divides it in multiple batches, reads the istantaneous frequency using the Hilbert transform (using the Cupy implementation), 
    recombines the results, writing them in a new npy file that will contain the istantaneous frequencies of the whole original file. 
    The division of the calculation in multiple batches is made to handle large files that would not fit in the computer memory. 
    The istantaneous frequency obtained from Hilbert transform give a large error at the edges of the array, so the division in multiple windows is done
    by taking a ovelapping region with the neighbour windows, that will be discarded before the results will be recombined. 
    
    Parameters:
    filename : str 
        Address of the npy file containing the data.
    output_length_batch : int
        Size of the batches that will be written on file.
    overlap_length : int
        Size of the portion of batch that overlaps with the neighbour batch (i.e. has the same elements). \
        The Hilbert transform are done on batches of size length_batch = output_length_batch + 2*overlap_length, then the overlap_length points
        at the beginning and the end are thrown away, and on file is written a batch of size output_length_batch.   
    Returns : 
    None
    The module writes the results on a npy file a ( np.floor(imported_data_length/output_length_batch)*output_length_batch , 2) ndarray. 
    The overlap_length must be > 1, and the output_length_batch must be >=1.
    If the batch length set is larger than the imported file, then the program takes the istantaneous frequencies from a single Hilbert transform of the whole file.
    """

    # check that the input values respect the program conditions
    if overlap_length <= 1:
        raise(ValueError("overlap_length must be an integer > 1") )
    if output_length_batch <= 0:
        raise(ValueError("output_length_batch must be a positive integer") )
    if filename.endswith(".npy") == False:
        raise(ImportError("The file to be imported must be a numpy (.npy) file") )

    current_batch_result = cp.empty((output_length_batch ,2)) #this cupy.ndarray will contain the result of the single batch
    imported_data = np.load(filename, mmap_mode='r') #nmap_mode='r' is set for large files
    imported_data_length = len(imported_data)
    if imported_data_length <= 0:
        raise(ValueError("The file contains no data"))
    length_batch = output_length_batch + 2*overlap_length  
    # This is the size of the array on which the Hilbert transform will be computed
    # (overlap_length points at the beginning of the batch, and overlap_length points at the end of it)
    if output_length_batch > imported_data_length:  
        # if the requested batch length is larger than the file, 
        # then the module performs a unique Hilbert transform on the whole file. 
        # So here the output_length_batch and the current_batch_resuls are recalculated
        length_batch = imported_data_length
        output_length_batch = length_batch - 2*overlap_length
        current_batch_result = current_batch_result[:output_length_batch,:]
    # recalculate the filename of the output file with the parameters used
    # (it is recalculated here because for output_length_batch > imported_data_length the output_length_batch does not match with the value given by the user)
    filestem = PurePath(filename).stem
    name_format = "_frequency-output_length_batch={bat:d}-overlap={over:d}.npy"
    output_name = name_format.format(bat = output_length_batch,over = overlap_length)
    output_filename = PurePath(filename).with_name(filestem + output_name) 
    # the data are rearranged in a 2D array where each row is a batch (see respective module description commment for more details) 
    slided_array = slide_array_overlap(imported_data,length_batch,overlap_length)
    with NpyAppendArray(output_filename, delete_if_exists=True) as npaa:
        for batch in tqdm(slided_array): # loop over the rows of the slided_array (batches) with time profiling (tqdm)
            time = batch[0]
            wave = batch[1]
            time = cp.asarray(time)
            wave = cp.asarray(wave)
            SR = 1/(time[1] - time[0])
            hilbert_result = istantaneous_frequency_hilbert(wave,SR)
            time = time[overlap_length  : -overlap_length]    #here the initial and final part of the batch that overlap with the neighbour batch are rejected
            frequency = hilbert_result[overlap_length  : -overlap_length+1] # the read_signal_frequency_hilbert uses np.diff, which returns N-1 elements  
            current_batch_result[:,0] = time
            current_batch_result[:,1] = frequency
            npaa.append(current_batch_result) #appends on the file "output_filename" the calculated frequency
        print(output_filename,"completed!")

def istantaneous_frequency_hilbert(signal : cp.array ,sampling_rate : cp.float64):
    """
    Calculates the istantaneous frequency of "signal" of shape (N,) sampled with 'sampling_rate' frequency using the Hilbert transform from cupyx.scipy.signal.
    It returns a cupy.ndarray of shape (N-1,) containing the istantaneous frequencies.
    """
    analytic_signal = cupy_hilbert(signal)
    instantaneous_phase = cp.unwrap(cp.angle(analytic_signal))  
    instantaneous_frequency = cp.diff(instantaneous_phase) / (2.0*cp.pi) * sampling_rate #dphase/dt = frequency
    return instantaneous_frequency

def slide_array_overlap(array : np.array, length_window : int, overlap_length: int):
    """
    Takes a numpy array ("array") of shape (N,), and returns a ndarray of shape (length_window , np.floor((N-overlap_length)/(length_window-overlap_length)) + 1).
    Each row of this array has its first and last "overlap_length" elements in common with the previous and following row.
    Example, with the following arguments: 
    a = np.array([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19])
    length_window = 6
    overlap_length = 1
    it returns:
array([[ 0,  1,  2,  3,  4,  5],
       [ 4,  5,  6,  7,  8,  9],
       [ 8,  9, 10, 11, 12, 13],
       [12, 13, 14, 15, 16, 17]])

    the overlap between two neighbour batches is 2*overlap_length so that by removing overlap_length on each 
    side of each batch the match is precise (in the example, by removing the first and last point on two rows
    you get the right sequence of integers).
    Note that some elements at the end of the array might be thrown away depending on the input parameters.
    """
    overlap_length_before_after = 2*overlap_length
    return sliding_window_view(array, length_window,axis=0)[::length_window - overlap_length_before_after]

#########################################################################################################

if __name__ == "__main__":
    name = "/long_decay_test_file.npy"
    dir_path = os.path.dirname(os.path.realpath(__file__))
    filename = dir_path + name
    output_length_batch = 1000
    overlap_length = 100
    hilbert_istantaneous_frequency_in_batches(filename, output_length_batch, overlap_length)