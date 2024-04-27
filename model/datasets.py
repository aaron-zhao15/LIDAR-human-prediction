import numpy as np
import copy
import torch
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, input_seqs, targets, seq_len, use_vel=True):
        self.use_vel = use_vel
        self.input_seqs = input_seqs
        self.targets = targets
        self.seq_len = seq_len

    def __len__(self):
        return len(self.input_seqs)

    def __getitem__(self, idx):
        input_seq = self.input_seqs[idx]
        target = self.targets[idx]
        # padding
        if len(input_seq) < self.seq_len:
            n = self.seq_len-len(input_seq)
            padding = torch.tile(input_seq[0], (n,1))
            input_seq = torch.cat((padding, input_seq), dim=0)
        if len(input_seq) > self.seq_len:
            # input_seq = input_seq[-self.seq_len:, ...]
            input_seq = input_seq[0:self.seq_len, ...]
        return input_seq, target
    
class TrajectoryWithSeqTaskDataset(Dataset):
    def __init__(self, input_seqs, target_seqs, task_labels, seq_len=60, num_classes=17, use_vel=False):
        self.use_vel = use_vel
        self.input_seqs = input_seqs
        self.target_seqs = target_seqs
        self.task_labels = task_labels
        self.seq_len = seq_len
        self.num_classes = num_classes

    def __len__(self):
        return len(self.input_seqs)

    def __getitem__(self, idx):
        input_seq = self.input_seqs[idx]
        target_seq = self.target_seqs[idx]
        task = self.task_labels[idx]
        # padding
        if len(input_seq) < self.seq_len:
            n = self.seq_len-len(input_seq)
            padding = torch.tile(input_seq[0], (n,1))
            input_seq = torch.cat((padding, input_seq), dim=0)
        if len(input_seq) > self.seq_len:
            # input_seq = input_seq[-self.seq_len:, ...]
            input_seq = input_seq[0:self.seq_len, ...]
        if len(target_seq) < self.seq_len:
            n = self.seq_len-len(target_seq)
            padding = torch.tile(target_seq[0], (n,1))
            target_seq = torch.cat((padding, target_seq), dim=0)
        if len(target_seq) > self.seq_len:
            # input_seq = input_seq[-self.seq_len:, ...]
            target_seq = target_seq[0:self.seq_len, ...]
        task = torch.nn.functional.one_hot(torch.tensor(int(task)), self.num_classes)
        return input_seq, target_seq, task


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
