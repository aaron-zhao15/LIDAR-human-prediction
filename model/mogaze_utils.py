# Python program to demonstrate
# HDF5 file
 
import numpy as np
import h5py
import glob

def read_from_hdf(hdf_path):
    """
    Read the data from a .hdf5 file into a numpy array and return it.
    @hdf_path: The string pathname of the specified .hdf5 file.
    """
    with h5py.File(hdf_path, 'r') as f:
        group_keys = list(f.keys())
        items = f.attrs.items()
        # print(f['data'].attrs['description'])
        data = f.get(group_keys[0])
        return np.array(data)
    
def read_from_csv(csv_path):
    """
    Read the data from a .txt file into a numpy array and return it.
    @csv_path: The string pathname of the specified .txt file.
    """
    return np.genfromtxt(csv_path, delimiter=',')

def read_hdf_from_folder(folder_path="../mogaze_data/"):
    """
    Read the data from a folder containing .hdf5 files into a list of numpy arrays and return it.
    @folder_path: The string pathname of the folder. Ends in /
    """
    human_data_paths = glob.glob(folder_path + "*human_data.hdf5")
    data_set = [read_from_hdf(path) for path in human_data_paths]
    return data_set

def read_csv_from_folder(folder_path="../mogaze_data/"):
    """
    Read the data from a folder containing .hdf5 files into a list of numpy arrays and return it.
    @folder_path: The string pathname of the folder. Ends in /
    """
    human_data_paths = glob.glob(folder_path + "*human_data.hdf5")
    data_set = [read_from_hdf(path) for path in human_data_paths]
    return data_set


def get_velocities(joint_positions, dt=1/120):
    if type(joint_positions) == np.ndarray:
        return (joint_positions[1:]-joint_positions[:-1])/dt
    else:
        return [(data[1:]-data[:-1])/dt for data in joint_positions]

def downsample_data(dataset, frequency=20):
    """
    Avoid using this method, use step size in sequence_from_array to perform downsampling.
    Take as input, either a numpy array or a list of numpy arrays and downsample to every 20 datapoints.
    @dataset: The dataset, either on its own or a list of other datasets.
    @frequency: Frequency of sampling points. Units are frames/second.
    """
    assert type(dataset) == list or type(dataset) == np.ndarray
    if type(dataset) == np.ndarray:
        return dataset[::frequency]
    else:
        return [data[::frequency] for data in dataset]


def sequence_from_array(data_array, seq_len, target_offset, step_size=20):
    """
    For implementation, avoid using this method and instead use sequences_from_framedata for consistent type handling.

    From a single array of data, parse the sequential data into tuples of sequences, (input_seq, target_seq).
    There's an assumption made, that the input and target sequence lengths are both equal. If the use case
    requires a different length for each sequence, a different method needs to be used.
    @joint_positions: The data in the form of a numpy ndarray.
    @seq_len: The length of the input and target sequences.
    @target_offset: How far the target sequence is shifted from the start of the input sequence. You can think
    about this as how many time steps into the future are we modeling.
    @step_size: step size is used to specify the frequency of sampling.
    """
    input_sequences, target_sequences = [], []
    for start_index in range(len(data_array)-((seq_len-1)+target_offset)*step_size):
        input_sequence = [data_array[start_index+(i)*step_size] for i in range(seq_len)]
        # the assumption here is that the input and target sequence lengths are equal
        target_sequence = [data_array[start_index+(i+target_offset)*step_size] for i in range(seq_len)]
        
        input_sequence = np.array(input_sequence)
        target_sequence = np.array(target_sequence)

        input_sequences.append(input_sequence)
        target_sequences.append(target_sequence)
    return [input_sequences, target_sequences]

def sequences_from_framedata(dataset, seq_len, target_offset=3):
    """
    Generalized form of sequence_from_array, which can handle both a singular data array or a list of data 
    arrays. The important part is this method returns the input and target sequences in list form.
    """
    assert type(dataset) == list or type(dataset) == np.ndarray
    if type(dataset) == np.ndarray:
        return [sequence_from_array(dataset, seq_len, target_offset)]
    else:
        return [sequence_from_array(data, seq_len, target_offset) for data in dataset]

def write_seq_to_file(array_list, file_path="/Users/aaronzhao/human_prediction/LIDAR-human-prediction/mogaze_data/sequences/"):
    """
    Write a list of input-target sequence pairs to some file path. This method doesn't preserve info about the
    original name identifier, so it should really only be used for temporary data storage. Serves as a kind of
    cache. 
    """
    return

def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    normalized = (data-mean)/std
    return normalized, (mean, std)



def sanity_check():
    # dataset = read_from_folder()
    # assert type(dataset) == list
    # assert type(dataset[0]) == np.ndarray

    joint_posns = read_from_hdf("../mogaze_data/p1_1_human_data.hdf5")
    joint_vels = get_velocities(joint_posns)
    print(len(joint_vels))
    print(len(joint_posns))
    normalized_posns, _ = normalize(joint_posns)
    normalized_vels, _ = normalize(joint_vels)
    print(normalized_posns.mean(), normalized_posns.std())
    print(normalized_vels.mean(), normalized_vels.std())

    [input_sequence, target_sequence] = sequence_from_array(joint_posns, 4, 3)
    print(np.array(input_sequence).shape)
    print(np.array(target_sequence).shape)
    print(input_sequence[3*20] == target_sequence[0])
    print(joint_posns[0])

# sanity_check()






