from torch import from_numpy
from torch.utils.data import Dataset
from random import seed
from random import sample
from src.utils import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler


def add_timepoint_embedding(input_df, timepoint_label):
    current_timepoint_df = pd.get_dummies(input_df.index.get_level_values('label'))
    for col in timepoint_label:
        if col not in current_timepoint_df.columns:
            current_timepoint_df[col] = False
    current_timepoint_df = current_timepoint_df.astype(input_df.iloc[:, 1].dtype)
    current_timepoint_df = current_timepoint_df[timepoint_label]
    current_timepoint_df.index = input_df.index
    # output_df = pd.concat([current_timepoint_df, input_df], axis=1)
    return current_timepoint_df


def load_split_data(data_dir, missTimepoint, valSet_ratio, trainNum, obsNum,
                    set_seed, save_data_dir=None):
    trainSplit_raw = dict()
    validSplit_raw = dict()
    testSplit_raw = dict()

    train_idx = None
    valid_idx = None
    timepoint_label = None
    missTimepoint = [missTimepoint]

    if save_data_dir is None:
        save_idx = False
    else:
        save_idx = True

    seed(set_seed)
    print("+ loading data:")
    for data_part in ["vA", "vB"]:
        print("  -", data_part)
        ### load data for training and validation
        dataTrainVal = pd.read_csv(os.path.join(data_dir, data_part + "_train" + ".csv", ),
                                   index_col=["label", "zz_nr"])
        if timepoint_label is None:
            timepoint_label = pd.get_dummies(dataTrainVal.index.get_level_values('label')).columns
        dataTrainVal_time_label = add_timepoint_embedding(input_df=dataTrainVal, timepoint_label=timepoint_label)

        ### load test data
        dataTest = pd.read_csv(os.path.join(data_dir, data_part + "_test" + ".csv", ),
                               index_col=["label", "zz_nr"])
        dataTest_time_label = add_timepoint_embedding(input_df=dataTest, timepoint_label=timepoint_label)
        # testSplit_raw[data_part] = dict(tuple(dataTest.groupby(level="label")))

        testSplit_raw[data_part] = {
            "data": dataTest,
            "time_label": dataTest_time_label
        }

        ### prepare train set and validation set
        if valid_idx is None:
            sample_id = dataTrainVal.index.unique(level='zz_nr')
            valid_idx = np.random.choice(sample_id, size=int(len(sample_id) * valSet_ratio), replace=False)

            if save_idx:
                pd.DataFrame(valid_idx, columns=['zz_nr']).to_csv(os.path.join(save_data_dir, "valSet_idx_trainNum_" +
                                                                               str(trainNum) + "_obsNum_" + str(
                    obsNum) + "_seed_" + str(set_seed) + ".csv"), index=False)

        valid_mask = dataTrainVal.index.get_level_values('zz_nr').isin(valid_idx)
        validSplit_raw[data_part] = {
            "data": dataTrainVal[valid_mask],
            "time_label": dataTrainVal_time_label[valid_mask]
        }
        # validSplit_raw[data_part] = dict(tuple(dataVal.groupby(level="label")))

        dataTrain = {
            "data": dataTrainVal[~valid_mask],
            "time_label": dataTrainVal_time_label[~valid_mask]
        }
        if trainNum == "all":
            trainSplit_raw[data_part] = dataTrain
        elif isinstance(trainNum, int):
            if train_idx is None:
                if 0 < trainNum <= dataTrain["data"].shape[0]:
                    train_idx = np.random.choice(dataTrain["data"].index.get_level_values('zz_nr').unique(),
                                                 size=trainNum, replace=False)
                    train_mask = dataTrain["data"].index.get_level_values('zz_nr').isin(train_idx)
                    trainSplit_raw[data_part] = {
                        "data": dataTrain["data"][train_mask],
                        "time_label": dataTrain["time_label"][train_mask]
                    }
                    if save_idx:
                        pd.DataFrame(train_idx, columns=['zz_nr']).to_csv(
                            os.path.join(save_data_dir, "trainSet_idx_trainNum_" + str(trainNum) +
                                         "_obsNum_" + str(obsNum) + "_seed_" + str(set_seed) + ".csv"), index=False
                        )
                else:
                    raise ExceptionObsNum("trainNum", "> 0", trainNum, dataTrain["data"].shape[0])
            trainSplit_raw[data_part] = dataTrain.iloc[train_idx]
        else:
            raise ValueError("trainNum should be 'all' or an integer!")

    # prepare observation map
    observed_row = ~trainSplit_raw["vB"]["data"].isna().all(axis=1).values
    if obsNum == "auto":
        trainSplit_raw["observation_map"] = observed_row.reshape(-1, 1)
    elif isinstance(obsNum, int):
        observed_row_missingTimepoint = np.where(np.logical_and(
            observed_row,
            trainSplit_raw["vB"]["data"].index.get_level_values('label').isin(missTimepoint)
        ))[0]
        if 0 <= obsNum <= len(observed_row_missingTimepoint):
            obsNum_idx = np.random.choice(observed_row_missingTimepoint, size=obsNum, replace=False)
            obsNum_row = (
                (observed_row) &
                (~trainSplit_raw["vB"]["data"].index.get_level_values('label').isin(missTimepoint))
            ).reshape(-1, 1)
            obsNum_row[obsNum_idx, 0] = True

            trainSplit_raw["observation_map"] = obsNum_row

            if save_idx:
                pd.DataFrame(obsNum_row).to_csv(
                    os.path.join(save_data_dir, "obsNum_idx_trainNum_" + str(trainNum) +
                                 "_obsNum_" + str(obsNum) + "_seed_" + str(set_seed) + ".csv"), index=False
                )
        else:
            raise ExceptionObsNum("obsNum", ">= 0", obsNum, len(observed_row_missingTimepoint))
    else:
        raise ValueError("obsNum should be 'auto' or an integer!")

    validSplit_raw["observation_map"] = ~validSplit_raw["vB"]["data"].isna().all(axis=1).values.reshape(-1, 1)
    testSplit_raw["observation_map"] = ~testSplit_raw["vB"]["data"].isna().all(axis=1).values.reshape(-1, 1)

    return {
        'trainSplit_raw': trainSplit_raw,
        'validSplit_raw': validSplit_raw,
        'testSplit_raw': testSplit_raw
    }


def scale_data(dataSplit_raw_dict, use_scaler):
    trainSplit_raw = dataSplit_raw_dict["trainSplit_raw"]
    validSplit_raw = dataSplit_raw_dict["validSplit_raw"]
    testSplit_raw = dataSplit_raw_dict["testSplit_raw"]

    if use_scaler == "standard":
        data_scaler = StandardScaler
    elif use_scaler == "robust":
        data_scaler = RobustScaler
    elif use_scaler == "minmax":
        data_scaler = MinMaxScaler
    else:
        raise Exception('scale should be one of "minmax", "robust", or "standard"!')
    print("+ Data scaling using " + use_scaler + " scaler:")

    scaler_vA = data_scaler().fit(trainSplit_raw["vA"]["data"])
    scaler_vB = data_scaler().fit(trainSplit_raw["vB"]["data"].iloc[trainSplit_raw['observation_map']])

    trainSplit_scaled = trainSplit_raw.copy()
    validSplit_scaled = validSplit_raw.copy()
    testSplit_scaled = testSplit_raw.copy()

    for data_part in ["vA", "vB"]:
        if data_part == "vA":
            print("  -", data_part + ": use scaler_vA")
            use_this_scaler = scaler_vA
        else:
            print("  -", data_part + ": use scaler_vB")
            use_this_scaler = scaler_vB
        trainSplit_scaled[data_part]["data"] = pd.DataFrame(
            use_this_scaler.transform(trainSplit_raw[data_part]["data"]),
            columns=trainSplit_raw[data_part]["data"].columns,
            index=trainSplit_raw[data_part]["data"].index)

        if validSplit_raw[data_part]["data"].shape[0] > 0:
            validSplit_scaled[data_part]["data"] = pd.DataFrame(
                use_this_scaler.transform(validSplit_raw[data_part]["data"]),
                columns=validSplit_raw[data_part]["data"].columns,
                index=validSplit_raw[data_part]["data"].index)
        else:
            validSplit_scaled = pd.DataFrame()

        testSplit_scaled[data_part]["data"] = pd.DataFrame(
            use_this_scaler.transform(testSplit_raw[data_part]["data"]),
            columns=testSplit_raw[data_part]["data"].columns,
            index=testSplit_raw[data_part]["data"].index)

        # trainSplit_scaled["mask_" + data_part] = trainSplit_raw["mask_" + data_part]
        # validSplit_scaled["mask_" + data_part] = validSplit_raw["mask_" + data_part]
        # testSplit_scaled["mask_" + data_part] = testSplit_raw["mask_" + data_part]

    return {
        'scaler_viewA': scaler_vA,
        'scaler_viewB': scaler_vB,
        'trainSplit_scaled': trainSplit_scaled,
        'validSplit_scaled': validSplit_scaled,
        'testSplit_scaled': testSplit_scaled
    }


class OmicsDataset(Dataset):
    def __init__(self, dataset_scaled):

        observation_map = pd.DataFrame(dataset_scaled['observation_map'],
                                       index=dataset_scaled['vB']['data'].index)
        self.observation_map = [group.to_numpy() for _, group in observation_map.groupby(level="label", sort=False)]
        all_observed_timePoint = np.array([all(_) for _ in self.observation_map])
        self.complete_timePoint = np.where(all_observed_timePoint)[0]
        print("Timepoints with complete data:", self.complete_timePoint)
        # data_viewA_df = pd.concat([dataset_scaled['vA']['time_label'], dataset_scaled['vA']['data']], axis=1)
        # data_viewB_df = pd.concat([dataset_scaled['vB']['time_label'], dataset_scaled['vB']['data']], axis=1)
        assert all(dataset_scaled['vA']['data'].index == dataset_scaled['vB']['data'].index), \
            "viewA and viewB should have the same index values!"

        # data_viewA = dict(tuple(data_viewA_df.groupby(level="label")))
        # data_viewB = dict(tuple(data_viewB_df.groupby(level="label")))

        # data_viewA_df = data_viewA_df.reset_index(level="zz_nr")
        # data_viewB_df = data_viewB_df.reset_index(level="zz_nr")

        self.data_viewA = [group.to_numpy() for _, group in dataset_scaled['vA']['data'].groupby(level="label", sort=False)]
        self.data_viewB = [group.to_numpy() for _, group in dataset_scaled['vB']['data'].groupby(level="label", sort=False)]

        self.label_viewA = [group.to_numpy() for _, group in dataset_scaled['vA']['time_label'].groupby(level="label", sort=False)]
        self.label_viewB = [group.to_numpy() for _, group in dataset_scaled['vB']['time_label'].groupby(level="label", sort=False)]

        assert all(dataset_scaled['vA']['time_label'].index == dataset_scaled['vB']['time_label'].index), \
            "viewA and viewB should have the same labels for timepoints!"
        # mask_viewA = ~data_viewA.isna()
        # mask_viewB = ~data_viewB.isna()
        self.load_different_timePoint()

    def load_different_timePoint(self):
        # only two timepoints are used in one training epoch
        # the first timepoint is always the fully observed data block
        self.first_timePoint = int(np.random.choice(self.complete_timePoint, 1))

        # the second timepoint can be complete or incomplete data block
        other_timePoint = [i for i in range(len(self.observation_map)) if i != self.first_timePoint]
        self.second_timePoint = int(np.random.choice(other_timePoint, 1))
        # print("Current timepoints indices:", self.first_timePoint, "&", self.second_timePoint)

        # self.current_observe_viewA_time1 = self.data_viewA[self.first_timePoint]
        # self.current_observe_viewB_time1 = self.data_viewB[self.first_timePoint]
        # self.current_observe_viewA_time2 = self.data_viewA[self.second_timePoint]
        # self.current_observe_viewB_time2 = self.data_viewB[self.second_timePoint]
        #
        # self.current_label_viewA_time1 = self.label_viewA[self.first_timePoint]
        # self.current_label_viewB_time1 = self.label_viewB[self.first_timePoint]
        # self.current_label_viewA_time2 = self.label_viewA[self.second_timePoint]
        # self.current_label_viewB_time2 = self.label_viewB[self.second_timePoint]

    def __len__(self):
        return len(self.observation_map[0])

    def __getitem__(self, idx):  # idx = [1,2,10, 45]

        observe_viewA_time1 = from_numpy(self.data_viewA[self.first_timePoint][idx, :].astype(np.float32))
        observe_viewB_time1 = from_numpy(self.data_viewB[self.first_timePoint][idx, :].astype(np.float32))
        observe_viewA_time2 = from_numpy(self.data_viewA[self.second_timePoint][idx, :].astype(np.float32))
        observe_viewB_time2 = from_numpy(self.data_viewB[self.second_timePoint][idx, :].astype(np.float32))

        label_viewA_time1 = from_numpy(self.label_viewA[self.first_timePoint][idx, :].astype(np.float32))
        label_viewB_time1 = from_numpy(self.label_viewB[self.first_timePoint][idx, :].astype(np.float32))
        label_viewA_time2 = from_numpy(self.label_viewA[self.second_timePoint][idx, :].astype(np.float32))
        label_viewB_time2 = from_numpy(self.label_viewB[self.second_timePoint][idx, :].astype(np.float32))

        # mask_viewA_time1 = from_numpy(self.mask_viewA_time1[idx, :].astype(np.float32))
        # mask_viewB_time1 = from_numpy(self.mask_viewB_time1[idx, :].astype(np.float32))
        # mask_viewA_time2 = from_numpy(self.mask_viewA_time2[idx, :].astype(np.float32))
        # mask_viewB_time2 = from_numpy(self.mask_viewB_time2[idx, :].astype(np.float32))

        observation_map = from_numpy(self.observation_map[self.second_timePoint][idx, :].astype(np.float32))

        data = {'batch_viewA_time1': observe_viewA_time1,
                'batch_viewB_time1': observe_viewB_time1,
                'batch_viewA_time2': observe_viewA_time2,
                'batch_viewB_time2': observe_viewB_time2,

                'label_viewA_time1': label_viewA_time1,
                'label_viewB_time1': label_viewB_time1,
                'label_viewA_time2': label_viewA_time2,
                'label_viewB_time2': label_viewB_time2,

                # 'mask_viewA_time1': mask_viewA_time1,
                # 'mask_viewB_time1': mask_viewB_time1,
                # 'mask_viewA_time2': mask_viewA_time2,
                # 'mask_viewB_time2': mask_viewB_time2,

                'observation_map': observation_map}

        return data


class IncompleteDataset(Dataset):
    def __init__(self, dataset_scaled, missTimepoint):
        assert all(dataset_scaled['vA']['data'].index == dataset_scaled['vB']['data'].index), \
            "viewA and viewB should have the same index values!"

        self.groundtruth = dataset_scaled['vB']['data'].loc[missTimepoint].to_numpy()
        self.data_viewA = dataset_scaled['vA']['data'].loc[missTimepoint].to_numpy()
        self.label_viewA = dataset_scaled['vA']['time_label'].loc[missTimepoint].to_numpy()

        timePoint_label = dataset_scaled['vB']['data'].index.unique('label')
        select_label_vB = timePoint_label[np.where(~timePoint_label.isin([missTimepoint]))[0][0]]
        self.data_viewB = dataset_scaled['vB']['data'].loc[select_label_vB].to_numpy()
        self.label_viewB = dataset_scaled['vB']['time_label'].loc[select_label_vB].to_numpy()

    def __len__(self):
        return self.data_viewA.shape[0]

    def __getitem__(self, idx):  # idx = [1,2,10, 45]

        observe_viewB_time1 = from_numpy(self.data_viewB[idx, :].astype(np.float32))
        observe_viewA_time2 = from_numpy(self.data_viewA[idx, :].astype(np.float32))

        label_viewB_time1 = from_numpy(self.label_viewB[idx, :].astype(np.float32))
        label_viewA_time2 = from_numpy(self.label_viewA[idx, :].astype(np.float32))

        observe_viewB_time2 = from_numpy(self.groundtruth[idx, :].astype(np.float32))

        data = {'batch_viewB_time1': observe_viewB_time1,
                'batch_viewA_time2': observe_viewA_time2,
                'batch_viewB_time2': observe_viewB_time2,

                'label_viewB_time1': label_viewB_time1,
                'label_viewA_time2': label_viewA_time2}

        return data


class EmbeddingDataset(Dataset):
    def __init__(self, dataset_scaled):

        self.data_viewA_embed = dataset_scaled['vA']['data'].to_numpy()
        self.label_viewA_embed = dataset_scaled['vA']['time_label'].to_numpy()

        self.data_viewB_embed = dataset_scaled['vB']['data'].to_numpy()
        self.label_viewB_embed = dataset_scaled['vB']['time_label'].to_numpy()

    def __len__(self):
        return self.data_viewA_embed.shape[0]

    def __getitem__(self, idx):  # idx = [1,2,10, 45]

        data_viewB_embed = from_numpy(self.data_viewB_embed[idx, :].astype(np.float32))
        data_viewA_embed = from_numpy(self.data_viewA_embed[idx, :].astype(np.float32))

        label_viewB_embed = from_numpy(self.label_viewB_embed[idx, :].astype(np.float32))
        label_viewA_embed = from_numpy(self.label_viewA_embed[idx, :].astype(np.float32))

        data = {'data_viewA': data_viewA_embed,
                'data_viewB': data_viewB_embed,

                'label_viewA': label_viewA_embed,
                'label_viewB': label_viewB_embed
                }

        return data


# data_dir="data/S4F4FF4"
# missTimepoint="FF4"
# valSet_ratio=0.2
# trainNum="all"
# obsNum=0
# use_scaler="standard"
# set_seed=1
# save_idx=False
def prepare_dataset(data_dir, missTimepoint,
                    valSet_ratio=0.2, trainNum="all",
                    obsNum=0, use_scaler="standard", set_seed=1, save_data_dir=None):
    dataSplit_raw = load_split_data(data_dir=data_dir, missTimepoint=missTimepoint,
                                    valSet_ratio=valSet_ratio, trainNum=trainNum, obsNum=obsNum,
                                    set_seed=set_seed, save_data_dir=save_data_dir)
    dataSplit_scaled = scale_data(dataSplit_raw_dict=dataSplit_raw, use_scaler=use_scaler)

    # dataset_scaled = dataSplit_scaled['validSplit_scaled']
    train_set = OmicsDataset(dataset_scaled=dataSplit_scaled['trainSplit_scaled'])
    if isinstance(dataSplit_scaled['validSplit_scaled'], dict): #dataSplit_scaled['validSplit_scaled'].shape[0] == 0:
        val_set = IncompleteDataset(dataset_scaled=dataSplit_scaled['validSplit_scaled'], missTimepoint=missTimepoint)
        embedding_set = EmbeddingDataset(dataset_scaled=dataSplit_scaled['validSplit_scaled'])
    else:
        val_set = IncompleteDataset(dataset_scaled=dataSplit_scaled['testSplit_scaled'], missTimepoint=missTimepoint)
        embedding_set = EmbeddingDataset(dataset_scaled=dataSplit_scaled['testSplit_scaled'])
    test_set = IncompleteDataset(dataset_scaled=dataSplit_scaled['testSplit_scaled'], missTimepoint=missTimepoint)

    output_dataset = {
        'train_set': train_set,
        'val_set': val_set,
        'test_set': test_set,
        'embedding_set': embedding_set,
        'scaler_viewA': dataSplit_scaled['scaler_viewA'],
        'scaler_viewB': dataSplit_scaled['scaler_viewB'],
        'dataSplit_raw': dataSplit_raw
    }

    return output_dataset

