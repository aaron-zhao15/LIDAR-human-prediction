import numpy as np
import copy
from torch.utils.data import Dataset

class TrajectoryDataset(Dataset):
    def __init__(self, input_seqs, input_vels, target_seqs, target_vels):
        self.input_seqs = input_seqs
        self.input_vels = input_vels
        self.target_seqs = target_seqs
        self.target_vels = target_vels
        self.input_data = np.append(input_seqs, input_vels, axis=2)
        self.target_data = np.append(target_seqs, target_vels, axis=2)

    def __len__(self):
        return len(self.input_vels)

    def __getitem__(self, idx):
        input = self.input_data[idx]
        # input = self.input_vels[idx]
        target = self.target_data[idx]
        # target = self.target_vels[idx]
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

        # decoder_inputs = data_sel[self._source_seq_len - 1:self._source_seq_len + self._target_seq_len - 1]
        # decoder_outputs = data_sel[self._source_seq_len:]
        # mirror for data augmentation
        # if random.random() > 0.5:
        #     data_sel = np.flip(data_sel, axis=0).copy()

        encoder_inputs = data_sel[0:self._source_seq_len - 1]
        decoder_target = data_sel[self._source_seq_len-1:]

        return encoder_inputs, decoder_target, action
