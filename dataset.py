import torch.utils.data as data
import scipy.io as scio
import numpy as np
import torch

class DatasetFrommatlab(data.Dataset):
    def __init__(self, data_path, target_path, input, target):
        super(DatasetFrommatlab, self).__init__()
        data_mat = scio.loadmat(data_path)
        # ATFRSD
        # self.data = (np.array(data_mat[input]) - 777.1)/(2086.0-777.1)
        # ATFRSD-W
        self.data = (np.array(data_mat[input]) - 640.8) / (1769.4 - 640.8)

        target_mat = scio.loadmat(target_path)
        # ATFRSD
        # self.target = (np.array(target_mat[target]) -777.1)/(2086.0-777.1)
        # ATFRSD-W
        self.target = (np.array(target_mat[target]) - 640.8) / (1769.4 - 640.8)


    def __getitem__(self, index):
        return torch.from_numpy(self.data[index, :, :, :]).float(), torch.from_numpy(self.target[index, :, :, :]).float()

    def __len__(self):
        return self.data.shape[0]
