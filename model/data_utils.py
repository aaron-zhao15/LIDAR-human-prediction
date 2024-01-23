# Python program to demonstrate
# HDF5 file
 
import numpy as np
import h5py
import glob
import pandas as pd
from TrajectoryDataset import TrajectoryDataset

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

def read_hdf_from_folder(folder_path="../humoro/mogaze/"):
    """
    Read the data from a folder containing .hdf5 files into a list of numpy arrays and return it.
    @folder_path: The string pathname of the folder. Ends in /
    """
    human_data_paths = glob.glob(folder_path + "*human_data.hdf5")
    data_set = [read_from_hdf(path) for path in human_data_paths]
    return data_set

def read_csv_from_folder(folder_path="../low_dim_data/"):
    """
    Read the data from a folder containing .hdf5 files into a list of numpy arrays and return it.
    @folder_path: The string pathname of the folder. Ends in /
    """
    human_data_paths = glob.glob(folder_path + "*.txt")
    data_set = [read_from_csv(path) for path in human_data_paths]
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
    return [np.array(input_sequences), np.array(target_sequences)]

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

def write_seqs_to_file(array_list, file_path="/Users/aaronzhao/human_prediction/LIDAR-human-prediction/data/sequences_collect"):
    """
    Write a list of input-target sequence pairs to some file path. This method doesn't preserve info about the
    original name identifier, so it should really only be used for temporary data storage. Serves as a kind of
    cache. 
    """
    combined_data = []
    for array in array_list:
        combined_data.extend(array)
    DF = pd.DataFrame(np.array(combined_data))
    DF.to_csv(file_path)
        

def hdf_to_txt(hdf_path):
    """
    Read the data from a .hdf5 file into a numpy array and return it.
    @hdf_path: The string pathname of the specified .hdf5 file.
    """
    with h5py.File(hdf_path, 'r') as f:
        group_keys = list(f.keys())
        items = f.attrs.items()
        # print(f['data'].attrs['description'])
        data = f.get(group_keys[0])
        DF = pd.DataFrame(np.array(data))
        DF.to_csv("data1.csv")



def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    normalized = (data-mean)/std
    return normalized, (mean, std)

def denormalize(normalized_data, mean, std):
    data = normalized_data*std
    data = data + mean
    return data

def generate_data_from_csv_folder(path, seq_len, target_offset, step_size, use_vel=True):
    joint_posns = read_csv_from_folder(path)
    input_seqs, target_seqs = [], []
    for joint_posn in joint_posns:
        j_posn = downsample_data(joint_posn)
        j_vel = get_velocities(j_posn, dt=step_size*(1/120))
        # j_posn, (j_posn_mean, j_posn_std) = normalize(j_posn)
        # j_vel, (j_vel_mean, j_vel_std) = normalize(j_vel)
        j_posn = j_posn[:-1]
        [i_seqs, t_seqs] = sequence_from_array(j_posn, seq_len, target_offset, step_size)
        [i_vel_seqs, t_vel_seqs] = sequence_from_array(j_vel, seq_len, target_offset, step_size)
        if use_vel:
            i_seqs = np.append(i_seqs, i_vel_seqs, axis=2)
            t_seqs = np.append(t_seqs, t_vel_seqs, axis=2)
        input_seqs.extend(i_seqs)
        target_seqs.extend(t_seqs)

    dataset = TrajectoryDataset(input_seqs, target_seqs, use_vel)
    return dataset

def generate_data_from_hdf_folder(path, seq_len, target_offset, step_size, use_vel=True):
    joint_posns = read_hdf_from_folder(path)
    input_seqs, target_seqs = [], []
    for joint_posn in joint_posns:
        j_posn = downsample_data(joint_posn)
        j_vel = get_velocities(j_posn, dt=step_size*(1/120))
        # j_posn, (j_posn_mean, j_posn_std) = normalize(j_posn)
        # j_vel, (j_vel_mean, j_vel_std) = normalize(j_vel)
        j_posn = j_posn[:-1]
        [i_seqs, t_seqs] = sequence_from_array(j_posn, seq_len, target_offset, step_size)
        [i_vel_seqs, t_vel_seqs] = sequence_from_array(j_vel, seq_len, target_offset, step_size)
        if use_vel:
            i_seqs = np.append(i_seqs, i_vel_seqs, axis=2)
            t_seqs = np.append(t_seqs, t_vel_seqs, axis=2)
        input_seqs.extend(i_seqs)
        target_seqs.extend(t_seqs)
    
    dataset = TrajectoryDataset(input_seqs, target_seqs, use_vel)
    return dataset

def sanity_check():
    # dataset = read_from_folder()
    # assert type(dataset) == list
    # assert type(dataset[0]) == np.ndarray

    joint_posns = read_from_hdf("../humoro/mogaze/p1_1_human_data.hdf5")
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
    
# generate_data_from_hdf_folder("../humoro/mogaze/", seq_len=50, target_offset=25, step_size=10)
# sanity_check()






