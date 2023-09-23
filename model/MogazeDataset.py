from torch.utils.data import Dataset

class MogazeDataset(Dataset):
    def __init__(self, input_seqs, target_seqs, input_vels, target_vels):
        self.input_seqs = input_seqs
        self.target_seqs = target_seqs
        self.input_vels = input_vels
        self.target_vels = target_vels

    def __len__(self):
        return len(self.input_vels)

    def __getitem__(self, idx):
        input = self.input_vels[idx]
        target = self.target_vels[idx]
        return input, target