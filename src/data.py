import numpy as np
from torch import from_numpy
from torch.utils.data import Dataset


class OmicsDataset(Dataset):
    def __init__(self, dataset_scaled):

        self.data_viewA_time1 = dataset_scaled['vA_t1']
        self.data_viewB_time1 = dataset_scaled['vB_t1']
        self.data_viewA_time2 = dataset_scaled['vA_t2']
        self.data_viewB_time2 = dataset_scaled['vB_t2']
        self.observation_map = dataset_scaled['observation_map']

        assert len(self.data_viewA_time1) == len(self.data_viewB_time1), \
            'data_viewA_time1 and data_viewA_time2 should have the same length!'

        assert len(self.data_viewA_time1) == len(self.data_viewA_time2), \
            'data_viewA_time1 and data_viewA_time2 should have the same length!'

        if self.data_viewB_time2 is not None:
            assert len(self.data_viewA_time1) == len(self.data_viewB_time2),\
                'data_viewA_time1 and data_viewB_time2 should have the same length!'

            # observation_map = np.ones((self.ata_viewB_time2.shape[0], 1), dtype=bool)
            # observation_map[np.where(np.isnan(self.data_viewB_time2[:, 0]))] = 0  # False/0 is missing
            # self.observation_map = observation_map
        else:
            self.observation_map = np.zeros((self.data_viewA_time1.shape[0], 1), dtype=bool)
            self.data_viewB_time2 = np.full(self.data_viewA_time1.shape, np.nan)

    def __len__(self):
        return len(self.data_viewA_time1)

    def __getitem__(self, idx):
        observe_viewA_time1 = from_numpy(self.data_viewA_time1[idx, :].astype(np.float32))
        observe_viewB_time1 = from_numpy(self.data_viewB_time1[idx, :].astype(np.float32))
        observe_viewA_time2 = from_numpy(self.data_viewA_time2[idx, :].astype(np.float32))
        observe_viewB_time2 = from_numpy(self.data_viewB_time2[idx, :].astype(np.float32))
        observation_map = from_numpy(self.observation_map[idx, :].astype(np.float32))

        data = {'batch_viewA_time1': observe_viewA_time1,
                'batch_viewB_time1': observe_viewB_time1,
                'batch_viewA_time2': observe_viewA_time2,
                'batch_viewB_time2': observe_viewB_time2,
                'observation_map': observation_map}

        return data
