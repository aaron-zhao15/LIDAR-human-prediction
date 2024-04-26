# Python program to demonstrate
# HDF5 file
 
import numpy as np
import h5py
import glob
# import pyarrow as pa
# import pandas as pd
from model.datasets import TrajectoryDataset, TrajectorySamplingDataset
# from datasets import TrajectoryDataset, TrajectorySamplingDataset
import csv
import copy
from pytorch3d.transforms import matrix_to_rotation_6d, rotation_6d_to_matrix
import torch
import math

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
        return torch.Tensor(np.array(data))
    
def read_from_csv(csv_path):
    """
    Read the data from a .txt file into a numpy array and return it.
    @csv_path: The string pathname of the specified .txt file.
    """
    # np_array = np.genfromtxt(csv_path, delimiter=',')
    np_array = []
    with open(csv_path, newline='') as f:
        reader = csv.reader(f)
        data = list(reader)
        try:
            np_array = np.asarray(data)
        except Exception as e:
            np_array = data
    return np_array

def read_hdf_from_folder(folder_path="../humoro/mogaze/"):
    """
    Read the data from a folder containing .hdf5 files into a list of numpy arrays and return it.
    @folder_path: The string pathname of the folder. Ends in /
    """
    human_data_paths = sorted(glob.glob(folder_path + "*_human_data.hdf5"))
    data_set = [read_from_hdf(path) for path in human_data_paths]
    return data_set

def read_csv_from_folder(folder_path="../humoro/mogaze/"):
    """
    Read the data from a folder containing .hdf5 files into a list of numpy arrays and return it.
    @folder_path: The string pathname of the folder. Ends in /
    """
    human_data_paths = sorted(glob.glob(folder_path + "*_instructions.csv"))
    data_set = [read_from_csv(path) for path in human_data_paths]
    return data_set


def get_velocities(joint_positions, dt=1/120):
    if type(joint_positions) == np.ndarray:
        return (joint_positions[int(120*dt):]-joint_positions[:int(-120*dt)])/dt
    else:
        return [(data[int(120*dt):]-data[:int(-120*dt)])/dt for data in joint_positions]

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
    for start_index in range(len(data_array)-(target_offset+2*(seq_len*step_size))):
        input_sequence = data_array[start_index:start_index+(seq_len*step_size):step_size]
        # the assumption here is that the input and target sequence lengths are equal
        target_sequence = data_array[start_index+target_offset+(seq_len*step_size):start_index+target_offset+2*(seq_len*step_size):step_size]

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

# def write_seqs_to_file(array_list, file_path="/Users/aaronzhao/human_prediction/LIDAR-human-prediction/data/sequences_collect"):
#     """
#     Write a list of input-target sequence pairs to some file path. This method doesn't preserve info about the
#     original name identifier, so it should really only be used for temporary data storage. Serves as a kind of
#     cache. 
#     """
#     combined_data = []
#     for array in array_list:
#         combined_data.extend(array)
#     DF = pd.DataFrame(np.array(combined_data))
#     DF.to_csv(file_path)
        

# def hdf_to_txt(hdf_path):
#     """
#     Read the data from a .hdf5 file into a numpy array and return it.
#     @hdf_path: The string pathname of the specified .hdf5 file.
#     """
#     with h5py.File(hdf_path, 'r') as f:
#         group_keys = list(f.keys())
#         items = f.attrs.items()
#         # print(f['data'].attrs['description'])
#         data = f.get(group_keys[0])
#         DF = pd.DataFrame(np.array(data))
#         DF.to_csv("data1.csv")



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
        # j_posn = downsample_data(joint_posn, step_size)
        j_posn = joint_posn
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

    dataset = TrajectoryDataset(input_seqs, target_seqs, seq_len, use_vel)
    return dataset

def generate_data_from_hdf_file(path, seq_len, target_offset, step_size, use_vel=True):
    joint_posns = [read_from_hdf(path)]
    input_seqs, target_seqs = [], []
    for joint_posn in joint_posns:
        # j_posn = downsample_data(joint_posn, step_size)
        j_posn = joint_posn
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
    
    dataset = TrajectoryDataset(input_seqs, target_seqs, seq_len, use_vel)
    return dataset

def generate_data_from_hdf_folder(path, seq_len, target_offset, step_size, use_vel=True):
    joint_posns = read_hdf_from_folder(path)
    input_seqs, target_seqs = [], []
    for joint_posn in joint_posns:
        # j_posn = downsample_data(joint_posn, step_size)
        j_posn = joint_posn
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
    
    dataset = TrajectoryDataset(input_seqs, target_seqs, seq_len, use_vel)
    return dataset

def generate_GT_data_from_hdf_file(path, seq_len, target_offset, step_size, use_vel=False):
    joint_posns = [read_from_hdf(path)]
    input_seqs, target_seqs = [], []
    for joint_posn in joint_posns:
        # j_posn = downsample_data(joint_posn, step_size)
        j_posn = joint_posn
        j_vel = get_velocities(j_posn, dt=step_size*(1/120))
        # j_posn, (j_posn_mean, j_posn_std) = normalize(j_posn)
        # j_vel, (j_vel_mean, j_vel_std) = normalize(j_vel)
        j_posn = j_posn[:-1]
        [i_seqs, t_seqs] = sequence_from_array(j_posn, seq_len, target_offset, step_size)
        [i_vel_seqs, t_vel_seqs] = sequence_from_array(j_vel, seq_len, target_offset, step_size)
        if use_vel:
            i_seqs = np.append(i_seqs, i_vel_seqs, axis=2)
            t_seqs = np.append(t_seqs, t_vel_seqs, axis=2)
        i_seqs = np.append(i_seqs, t_seqs, axis=1)[:, :-1, :]
        t_seqs = np.append(np.zeros((t_seqs.shape[0], t_seqs.shape[1]-1, t_seqs.shape[2])), t_seqs, axis=1)
        input_seqs.extend(i_seqs)
        target_seqs.extend(t_seqs)
    
    dataset = TrajectoryDataset(input_seqs, target_seqs, seq_len, use_vel)
    return dataset

def generate_GT_data_from_hdf_folder(path, seq_len, target_offset, step_size, use_vel=False):
    joint_posns = read_hdf_from_folder(path)
    input_seqs, target_seqs = [], []
    for joint_posn in joint_posns:
        # j_posn = downsample_data(joint_posn, step_size)
        j_posn = joint_posn
        j_vel = get_velocities(j_posn, dt=step_size*(1/120))
        # j_posn, (j_posn_mean, j_posn_std) = normalize(j_posn)
        # j_vel, (j_vel_mean, j_vel_std) = normalize(j_vel)
        j_posn = j_posn[:-1]
        [i_seqs, t_seqs] = sequence_from_array(j_posn, seq_len, target_offset, step_size)
        [i_vel_seqs, t_vel_seqs] = sequence_from_array(j_vel, seq_len, target_offset, step_size)
        if use_vel:
            i_seqs = np.append(i_seqs, i_vel_seqs, axis=2)
            t_seqs = np.append(t_seqs, t_vel_seqs, axis=2)
        i_seqs = np.append(i_seqs, t_seqs, axis=1)[:, :-1, :]
        t_seqs = np.append(np.zeros((t_seqs.shape[0], t_seqs.shape[1]-1, t_seqs.shape[2])), t_seqs, axis=1)
        input_seqs.extend(i_seqs)
        target_seqs.extend(t_seqs)
    
    dataset = TrajectoryDataset(input_seqs, target_seqs, seq_len, use_vel)
    return dataset

def generate_intent_segments_from_folder(person_path, seq_len=60, step_size=60, use_vel=False):
    joint_posns = read_hdf_from_folder(person_path)
    tasks_lst = read_csv_from_folder(person_path)
    input_seqs, targets = [], []
    for i, joint_posn in enumerate(joint_posns):
        tasks = tasks_lst[i]
        for j, task in enumerate(tasks):
            traj_start = int(task[0])
            traj_end = int(tasks[j+1][0]) if j < len(tasks)-1 else len(joint_posn)
            i_seq = copy.deepcopy(joint_posn[traj_start:traj_end:step_size])
            # i_seq = pose_6d_from_euler_angles(i_seq)
            input_seqs.append(i_seq)
            label_encoding = torch.nn.functional.one_hot(torch.tensor(int(task[1])), num_classes=17)
            targets.append(label_encoding)
    dataset = TrajectoryDataset(input_seqs, targets, seq_len, use_vel)
    return dataset

def generate_intent_data_from_person(person_path, sample_len=60, offset_len=60, step_size=1, use_vel=False):
    input_seqs = read_from_hdf(person_path+"_human_data.hdf5")
    tasks = read_from_csv(person_path+"_instructions.csv")
    targets = torch.zeros(input_seqs.shape[0])
    traj_start = 0
    for task in tasks:
        traj_end = int(task[0])
        targets[traj_start:traj_end] = int(task[1])
        traj_start = traj_end
    dataset = TrajectorySamplingDataset(input_seqs, targets, sample_len, offset_len, step_size, use_vel)
    return dataset

def generate_seq_to_seq(path, seq_len, target_offset, step_size, use_vel=False):
    joint_posns = read_hdf_from_folder(path)
    input_seqs, target_seqs = [], []
    for joint_posn in joint_posns:
        # j_posn = downsample_data(joint_posn, step_size)
        
        j_posn = pose_6d_from_euler_angles(joint_posn)
        j_vel = get_velocities(j_posn, dt=step_size*(1/120))
        # j_posn, (j_posn_mean, j_posn_std) = normalize(j_posn)
        # j_vel, (j_vel_mean, j_vel_std) = normalize(j_vel)
        j_posn = j_posn[:-1]
        [i_seqs, t_seqs] = sequence_from_array(j_posn, seq_len, target_offset, step_size)
        [i_vel_seqs, t_vel_seqs] = sequence_from_array(j_vel, seq_len, target_offset, step_size)
        if use_vel:
            i_seqs = np.append(i_seqs, i_vel_seqs, axis=2)
            t_seqs = np.append(t_seqs, t_vel_seqs, axis=2)
        i_seqs = np.append(i_seqs, t_seqs, axis=1)[:, :-1, :]
        t_seqs = np.append(np.zeros((t_seqs.shape[0], t_seqs.shape[1]-1, t_seqs.shape[2])), t_seqs, axis=1)
        input_seqs.extend(i_seqs)
        target_seqs.extend(t_seqs)
    
    dataset = TrajectoryDataset(input_seqs, target_seqs, seq_len, use_vel)
    return dataset

def euler_xyz_to_rotation_matrix(angles):
    # :param angles must be Nx3
    angles = angles.T
    def R_y(thetas):
        return torch.Tensor([[[torch.cos(theta), 0, torch.sin(theta)],
                              [0, 1, 0],
                              [-torch.sin(theta), 0, torch.cos(theta)]] for theta in thetas])
    def R_z(thetas):
        return torch.Tensor([[[torch.cos(theta), -torch.sin(theta), 0],
                              [torch.sin(theta), torch.cos(theta), 0],
                              [0, 0, 1]] for theta in thetas])
    def R_x(thetas):
        return torch.Tensor([[[1, 0, 0],
                              [0, torch.cos(theta), -torch.sin(theta)],
                              [0, torch.sin(theta), torch.cos(theta)]] for theta in thetas])
    X, Y, Z = R_x(angles[0]), R_y(angles[1]), R_z(angles[2])
    return Z@Y@X

def euler_angles_from_matrix(matrices):
    # https://eecs.qmul.ac.uk/~gslabaugh/publications/euler.pdf
    # assume matrices to be Nx3x3
    angle_list = []
    for R in matrices:
        angles = []
        if R[2, 0] != 1 and R[2, 0] != -1:
            theta1 = -torch.asin(R[2, 0])
            theta2 = torch.pi - theta1
            psi1 = torch.atan2(R[2, 1] / torch.cos(theta1), R[2, 2] / torch.cos(theta1))
            psi2 = torch.atan2(R[2, 1] / torch.cos(theta2), R[2, 2] / torch.cos(theta2))
            phi1 = torch.atan2(R[1, 0] / torch.cos(theta1), R[0, 0] / torch.cos(theta1))
            phi2 = torch.atan2(R[1, 0] / torch.cos(theta2), R[0, 0] / torch.cos(theta2))
            angles, angles2 = [psi1, theta1, phi1], [psi2, theta2, phi2]
        else:
            phi = torch.tensor(0.0)  # You can set phi to any value
            if R[2, 0] == -1:
                theta = torch.pi / 2
                psi = phi + torch.atan2(R[0, 1], R[0, 2])
            else:
                theta = -torch.pi / 2
                psi = -phi + torch.atan2(-R[0, 1], -R[0, 2])
            angles = [psi, theta, phi]
        angle_list.append(angles)
    return torch.Tensor(angle_list)

def euler_from_matrix(matrices, axes='sxyz'):
    # Return Euler angles from rotation matrix for specified axis sequence.
    #
    # axes : One of 24 axis sequences as string or encoded tuple
    #
    # Note that many Euler angle triplets can describe one matrix.
    #
    # >>> R0 = euler_matrix(1, 2, 3, 'syxz')
    # >>> al, be, ga = euler_from_matrix(R0, 'syxz')
    # >>> R1 = euler_matrix(al, be, ga, 'syxz')
    # >>> np.allclose(R0, R1)
    # True
    # >>> angles = (4.0*math.pi) * (np.random.random(3) - 0.5)
    # >>> for axes in _AXES2TUPLE.keys():
    # ...    R0 = euler_matrix(axes=axes, *angles)
    # ...    R1 = euler_matrix(axes=axes, *euler_from_matrix(R0, axes))
    # ...    if not np.allclose(R0, R1): print axes, "failed"
    _NEXT_AXIS = [1, 2, 0, 1]
    _EPS = torch.finfo(float).eps * 4.0
    _AXES2TUPLE = {
        'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
        'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
        'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
        'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
        'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
        'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
        'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
        'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        firstaxis, parity, repetition, frame = axes

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]
    
    angle_list = []
    for matrix in matrices:
        M = torch.Tensor(matrix)[:3, :3]
        if repetition:
            sy = torch.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
            if sy > _EPS:
                ax = torch.atan2(M[i, j], M[i, k])
                ay = torch.atan2(sy, M[i, i])
                az = torch.atan2(M[j, i], -M[k, i])
            else:
                ax = torch.atan2(-M[j, k], M[j, j])
                ay = torch.atan2(sy, M[i, i])
                az = 0.0
        else:
            cy = torch.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
            if cy > _EPS:
                ax = torch.atan2(M[k, j], M[k, k])
                ay = torch.atan2(-M[k, i], cy)
                az = torch.atan2(M[j, i], M[i, i])
            else:
                ax = torch.atan2(-M[j, k], M[j, j])
                ay = torch.atan2(-M[k, i], cy)
                az = 0.0

        if parity:
            ax, ay, az = -ax, -ay, -az
        if frame:
            ax, az = az, ax
        angle_list.append([ax, ay, az])
    return torch.Tensor(angle_list)

def pose_6d_from_euler_angles(joint_angles):
    # joint angles are assumed to be Nx66, so we'll reshape to 3x22xN
    joint_angles_reshaped = joint_angles.reshape((-1, 22, 3))
    base_translation = joint_angles_reshaped[:, 0, :]
    euler_angles = joint_angles_reshaped[:, 1:, :]

    joint_mats = [euler_xyz_to_rotation_matrix(euler_angles[:, i, :]) for i in range(euler_angles.shape[1])]
    continuous_6d = [matrix_to_rotation_6d(joint_mat) for joint_mat in joint_mats]
    collected_6d = torch.cat(continuous_6d, dim=1)
    pose_6d = torch.cat((base_translation, collected_6d), dim=1)
    return pose_6d

def euler_angles_from_pose_6d(base_with_pose_6d):
    # pose is assumed to be Nx129
    base_translation = base_with_pose_6d[:, 0:3]
    pose_6d = base_with_pose_6d[:, 3:].reshape((-1, 21, 6))
    rotation_mats = [rotation_6d_to_matrix(pose_6d[:, i, :]) for i in range(pose_6d.shape[1])]
    euler_angles = [euler_angles_from_matrix(matrix) for matrix in rotation_mats]
    euler_angles = torch.cat(euler_angles, dim=1)
    full_euler_angles = torch.cat((base_translation, euler_angles), dim=1)
    return full_euler_angles

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


# dataset = generate_intent_segments_from_folder("../humoro/mogaze/", seq_len=60, step_size=60)
# print([data[1] for data in dataset])

# joint_posns = read_from_hdf("../humoro/mogaze/p2_1_human_data.hdf5")[0:500, ...]
# pose_6d = pose_6d_from_euler_angles(joint_posns)
# print(pose_6d.shape)
# inv_euler = euler_angles_from_pose_6d(pose_6d)
# print(inv_euler.shape)
# print(torch.nn.functional.mse_loss(inv_euler, joint_posns))

# test_euler_angles = joint_posns[0:1, 3:6]
# print(test_euler_angles)
# test_rotation_mat = euler_xyz_to_rotation_matrix(test_euler_angles)
# print(test_rotation_mat.shape)
# inv_euler_angles = euler_angles_from_matrix(test_rotation_mat[0])
# print(inv_euler_angles[0])

# seq_len = 50
# target_offset = 50
# step_size = 20
# dataset = generate_data_from_hdf_file("../humoro/mogaze/p2_1_human_data.hdf5", seq_len, target_offset, step_size, use_vel=False)

# dataset = generate_intent_data_from_person("../humoro/mogaze/p1_1")

# generate_GT_data_from_hdf_file("../humoro/mogaze/p1_1_human_data.hdf5", seq_len=50, target_offset=50, step_size=1)


# sanity_check()






