import numpy as np
import copy
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, input_seqs, target_seqs, use_vel=True):
        self.use_vel = use_vel
        self.input_seqs = input_seqs
        self.target_seqs = target_seqs

    def __len__(self):
        return len(self.input_seqs)

    def __getitem__(self, idx):
        input = self.input_seqs[idx]
        # input = self.input_vels[idx]
        target = self.target_seqs[idx]
        # target = self.target_vels[idx]
        return input, target
    
class TrajectorySamplingDataset(Dataset):
    def __init__(self, input_trajectory, target_arr, step_size=1, sample_len=60, offset_len=60, use_vel=True):
        self.use_vel = use_vel
        self.input_trajectory = input_trajectory
        self.target_arr = target_arr
        self.step_size = step_size
        self.sample_len = sample_len
        self.offset_len = offset_len

    def __len__(self):
        return len(self.input_trajectory-self.sample_len*self.step_size)

    def __getitem__(self, idx):
        input = self.input_trajectory[idx:idx+self.sample_len:self.step_size, ...]
        target = self.target_arr[idx]
        return input, target


class generate_train_data(Dataset):
    def __init__(self, data_set, source_seq_len, target_seq_len, sample_start=16):
        self._data_set = data_set
        self._index_lst = list(data_set.keys())

        self._source_seq_len = source_seq_len
        self._target_seq_len = target_seq_len
        self._total_frames = self._source_seq_len + self._target_seq_len
        self._sample_start = sample_start
        self._action2index = self.get_dic_action2index()

    def __len__(self):
        return len(self._index_lst)

    def get_dic_action2index(self):
        actions = sorted(list(set([item[1] for item in self._data_set.keys()])) )
        return dict(zip(actions, range(0, len(actions))))

    def __getitem__(self, index):
        data = self._data_set[self._index_lst[index]]
        action = self._action2index.get(self._index_lst[index][1])

        # Sample somewherein the middle
        if data.shape[0] - self._total_frames <= 0:
            idx = 0
        else:
            idx = np.random.randint(self._sample_start, data.shape[0] - self._total_frames)
        # Select the data around the sampled points
        data_sel = copy.deepcopy(data[idx:idx + self._total_frames, :] )

        encoder_inputs = data_sel[0:self._source_seq_len - 1]
        decoder_target = data_sel[self._source_seq_len-1:]

        return encoder_inputs, decoder_target, action
