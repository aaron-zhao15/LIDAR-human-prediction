# Python program to demonstrate
# HDF5 file
 
import numpy as np
import h5py
import glob

def read_from_file(hdf_path):
    """
    Read the data from a .hdf5 file into a numpy array and return it.
    @hdf_path: The string pathname of the specified .hdf5 file.
    """
    with h5py.File(hdf_path, 'r') as f:
        group_keys = list(f.keys())
        # data = f.get('data')
        data = f.get(group_keys[0])
        return np.array(data)

def read_from_folder(folder_path="/Users/aaronzhao/human_prediction/LIDAR-human-prediction/mogaze/"):
    """
    Read the data from a folder containing .hdf5 files into a list of numpy arrays and return it.
    @folder_path: The string pathname of the folder.
    """
    human_data_paths = glob.glob(folder_path + "*human_data.hdf5")
    # print(human_data_paths)
    data_set = [read_from_file(path) for path in human_data_paths]
    return data_set


def downsample_data(dataset, frequency=20):
    """
    Take as input, either a numpy array or a list of numpy arrays and downsample to every 20 datapoints.
    @dataset: The dataset, either on its own or a list of other datasets.
    @frequency: Frequency of sampling points. Units are frames/second.
    """
    assert type(dataset) == list or type(dataset) == np.ndarray
    if type(dataset) == np.ndarray:
        return dataset[::frequency]
    else:
        return [data[::frequency] for data in dataset]


def sequence_from_array(array, seq_len, target_offset):
    """
    For implementation, avoid using this method and instead use sequences_from_framedata for consistent type handling.

    From a single array of data, parse the sequential data into tuples of sequences, (input_seq, target_seq).
    There's an assumption made, that the input and target sequence lengths are both equal. If the use case
    requires a different length for each sequence, a different method needs to be used.
    @array: The data in the form of a numpy ndarray.
    @seq_len: The length of the input and target sequences.
    @target_offset: How far the target sequence is shifted from the start of the input sequence. You can think
    about this as how many time steps into the future are we modeling.
    """
    input_sequences, target_sequences = [], []
    for step in range(len(array)-(seq_len-1)-target_offset):
        input_sequence = [array[step+i] for i in range(seq_len)]
        # the assumption here is that the input and target sequence lengths are equal
        target_sequence = [array[step+i+target_offset] for i in range(seq_len)]
        
        input_sequence = np.array(input_sequence).T
        target_sequence = np.array(target_sequence).T

        input_sequences.append(input_sequence)
        target_sequences.append(target_sequence)
    return [input_sequences, target_sequences]

def sequences_from_framedata(dataset, seq_len, target_offset=2):
    """
    Generalized form of sequence_from_array, which can handle both a singular data array or a list of data 
    arrays. The important part is this method returns the input and target sequences in list form.
    """
    assert type(dataset) == list or type(dataset) == np.ndarray
    if type(dataset) == np.ndarray:
        return [sequence_from_array(dataset, seq_len, target_offset)]
    else:
        return [sequence_from_array(data, seq_len, target_offset) for data in dataset]

def write_seq_to_file(array_list, file_path="/Users/aaronzhao/human_prediction/LIDAR-human-prediction/mogaze/sequences/"):
    """
    Write a list of input-target sequence pairs to some file path. This method doesn't preserve info about the
    original name identifier, so it should really only be used for temporary data storage. Serves as a kind of
    cache. 
    """
    return




def sanity_check():
    dataset = read_from_folder()
    assert type(dataset) == list
    assert type(dataset[0]) == np.ndarray

    data = downsample_data(dataset)
    data = np.array(sequences_from_framedata(data[0]))
    print(data.shape)

# sanity_check()






