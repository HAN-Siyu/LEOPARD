from torch import from_numpy
from torch.utils.data import Dataset
from random import seed
from random import sample
from src.utils import *
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler


def load_split_data(data_dir, valSet_ratio, trainNum, obsNum, set_seed,
                    save_idx=True, save_data_dir=None,
                    fill_na='mean', swap_time=False, swap_view=False):
    trainSplit_raw = dict()
    validSplit_raw = dict()
    testSplit_raw = dict()

    train_idx = None
    valid_idx = None

    print("+ loading data:")

    if swap_time:
        timeIdx_dict = {
            "t1": "t2",
            "t2": "t1"
        }
    else:
        timeIdx_dict = {
            "t1": "t1",
            "t2": "t2"
        }

    if swap_view:
        viewIdx_dict = {
            "vA": "vB",
            "vB": "vA"
        }
    else:
        viewIdx_dict = {
            "vA": "vA",
            "vB": "vB"
        }

    combined_dict = {}
    for view_key, view_value in viewIdx_dict.items():
        for time_key, time_value in timeIdx_dict.items():
            combined_key = f"{view_key}_{time_key}"
            combined_value = f"{view_value}_{time_value}"
            combined_dict[combined_key] = combined_value

    seed(set_seed)
    np.random.seed(set_seed)

    if save_data_dir is not None:
        save_folder_path = os.path.join(save_data_dir, "obsNum_" + str(obsNum),
                                        "seed_" + str(set_seed),
                                        )
        if not os.path.exists(save_folder_path):
            os.makedirs(save_folder_path)

    for data_part in combined_dict.keys():
        print("  -load", combined_dict[data_part] + " as", data_part)
        path_trainVal = os.path.join(data_dir, combined_dict[data_part] + "_train" + ".csv")
        path_test     = os.path.join(data_dir, combined_dict[data_part] + "_test" + ".csv")

        dataTrainVal = pd.read_csv(path_trainVal, index_col=["label", "zz_nr"])
        dataTest     = pd.read_csv(path_test,     index_col=["label", "zz_nr"])

        testSplit_raw[data_part] = dataTest

        if valid_idx is None:
            valid_idx = sample(range(dataTrainVal.shape[0]), int(dataTrainVal.shape[0] * valSet_ratio))
            if save_idx and save_data_dir is not None:
                dataTrainVal.iloc[valid_idx].index.get_level_values('zz_nr').to_frame(index=False).to_csv(
                    os.path.join(save_folder_path,
                                 "valSet_idx_trainNum_" + str(trainNum) +
                                 "_obsNum_" + str(obsNum) + ".csv"), index=False
                )

        validSplit_raw[data_part] = dataTrainVal.iloc[valid_idx].copy()

        dataTrain = dataTrainVal.drop(dataTrainVal.index[valid_idx], axis=0, inplace=False)

        if trainNum == "all":
            dataTrain_select = dataTrain
        elif isinstance(trainNum, int):
            if train_idx is None:
                if 0 < trainNum <= dataTrain.shape[0]:
                    train_idx = sample(range(dataTrain.shape[0]), trainNum)
                    if save_idx and save_data_dir is not None:
                        dataTrain.iloc[train_idx].index.get_level_values('zz_nr').to_frame(index=False).to_csv(
                            os.path.join(save_folder_path,
                                         "trainSet_idx_trainNum_" + str(trainNum) +
                                         "_obsNum_" + str(obsNum) + ".csv"), index=False
                        )
                else:
                    raise ExceptionObsNum("trainNum", "> 0", trainNum, dataTrain.shape[0])
            dataTrain_select = dataTrain.iloc[train_idx]
        else:
            raise ValueError("trainNum should be 'all' or an integer!")

        trainSplit_raw[data_part] = dataTrain_select

    trainSplit_raw["observation_map"] = np.zeros((trainSplit_raw["vB_t2"].shape[0], 1), dtype=bool)
    observed_idx_train = np.where(trainSplit_raw["vB_t2"].notna().any(axis=1).values)[0]

    if obsNum == "all":
        trainSplit_raw["observation_map"][observed_idx_train] = True
    elif isinstance(obsNum, int):
        if 0 <= obsNum <= len(observed_idx_train):
            obsNum_idx = sample(observed_idx_train.tolist(), obsNum)
            trainSplit_raw["observation_map"][obsNum_idx] = True
            if save_idx and save_data_dir is not None:
                obsNum_zzNr = trainSplit_raw["vB_t2"].iloc[trainSplit_raw["observation_map"]].index.get_level_values(
                    'zz_nr').to_frame(index=False)
                obsNum_zzNr.to_csv(
                    os.path.join(save_folder_path,
                                 "obsNum_idx_trainNum_" + str(trainNum) +
                                 "_obsNum_" + str(obsNum) + ".csv"), index=False
                )
        else:
            raise ExceptionObsNum("obsNum", ">= 0", obsNum, len(observed_idx_train))
    else:
        raise ValueError("obsNum should be 'all' or an integer!")

    validSplit_raw["observation_map"] = np.zeros((validSplit_raw["vB_t2"].shape[0], 1), dtype=bool)
    observed_idx_valid = np.where(validSplit_raw["vB_t2"].notna().any(axis=1).values)[0]
    validSplit_raw["observation_map"][observed_idx_valid] = True

    testSplit_raw["observation_map"] = np.zeros((testSplit_raw["vB_t2"].shape[0], 1), dtype=bool)
    observed_idx_test = np.where(testSplit_raw["vB_t2"].notna().any(axis=1).values)[0]
    testSplit_raw["observation_map"][observed_idx_test] = True

    for current_view_time in ["vA_t1", "vA_t2", "vB_t1", "vB_t2"]:
        trainSplit_raw[current_view_time].replace([np.inf, -np.inf], np.nan, inplace=True)
        validSplit_raw[current_view_time].replace([np.inf, -np.inf], np.nan, inplace=True)
        testSplit_raw[current_view_time].replace([np.inf, -np.inf], np.nan, inplace=True)

        trainSplit_raw["mask_" + current_view_time] = ~trainSplit_raw[current_view_time].isna()
        validSplit_raw["mask_" + current_view_time] = ~validSplit_raw[current_view_time].isna()
        testSplit_raw["mask_" + current_view_time] = ~testSplit_raw[current_view_time].isna()

        if fill_na == "mean":
            trainSplit_raw[current_view_time].fillna(trainSplit_raw[current_view_time].mean(), inplace=True)
            if current_view_time != "vB_t2":
                validSplit_raw[current_view_time].fillna(validSplit_raw[current_view_time].mean(), inplace=True)
                testSplit_raw[current_view_time].fillna(testSplit_raw[current_view_time].mean(), inplace=True)
        elif fill_na == "zero":
            trainSplit_raw[current_view_time].fillna(0, inplace=True)
            if current_view_time != "vB_t2":
                validSplit_raw[current_view_time].fillna(0, inplace=True)
                testSplit_raw[current_view_time].fillna(0, inplace=True)
        else:
            raise Exception("Wrong fillNA!")

    if not all(
            trainSplit_raw["vA_t1"].index.get_level_values('zz_nr') == trainSplit_raw["vA_t2"].index.get_level_values(
                'zz_nr')):
        raise Exception("Wrong zz_nr in train vA_t2")
    if not all(
            trainSplit_raw["vA_t1"].index.get_level_values('zz_nr') == trainSplit_raw["vB_t1"].index.get_level_values(
                'zz_nr')):
        raise Exception("Wrong zz_nr in train vB_t1")
    if not all(
            trainSplit_raw["vA_t1"].index.get_level_values('zz_nr') == trainSplit_raw["vB_t2"].index.get_level_values(
                'zz_nr')):
        raise Exception("Wrong zz_nr in train vB_t2")

    if not all(
            validSplit_raw["vA_t1"].index.get_level_values('zz_nr') == validSplit_raw["vA_t2"].index.get_level_values(
                'zz_nr')):
        raise Exception("Wrong zz_nr in valid vA_t2")
    if not all(
            validSplit_raw["vA_t1"].index.get_level_values('zz_nr') == validSplit_raw["vB_t1"].index.get_level_values(
                'zz_nr')):
        raise Exception("Wrong zz_nr in valid vB_t1")
    if not all(
            validSplit_raw["vA_t1"].index.get_level_values('zz_nr') == validSplit_raw["vB_t2"].index.get_level_values(
                'zz_nr')):
        raise Exception("Wrong zz_nr in valid vB_t2")

    if not all(testSplit_raw["vA_t1"].index.get_level_values('zz_nr') == testSplit_raw["vA_t2"].index.get_level_values(
            'zz_nr')):
        raise Exception("Wrong zz_nr in test vA_t2")
    if not all(testSplit_raw["vA_t1"].index.get_level_values('zz_nr') == testSplit_raw["vB_t1"].index.get_level_values(
            'zz_nr')):
        raise Exception("Wrong zz_nr in test vB_t1")
    if not all(testSplit_raw["vA_t1"].index.get_level_values('zz_nr') == testSplit_raw["vB_t2"].index.get_level_values(
            'zz_nr')):
        raise Exception("Wrong zz_nr in test vB_t2")

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

    vA_allTrain = pd.concat([trainSplit_raw['vA_t1'], trainSplit_raw['vA_t2']], axis=0)
    vB_allTrain = pd.concat([trainSplit_raw['vB_t1'],
                             trainSplit_raw['vB_t2'].iloc[trainSplit_raw['observation_map']]],
                            axis=0)

    scaler_vA = data_scaler().fit(vA_allTrain.to_numpy())
    scaler_vB = data_scaler().fit(vB_allTrain.to_numpy())

    trainSplit_scaled = {"observation_map": trainSplit_raw["observation_map"]}
    validSplit_scaled = {"observation_map": validSplit_raw["observation_map"]}
    testSplit_scaled = {"observation_map": testSplit_raw["observation_map"]}

    for data_part in ["vA_t1", "vA_t2", "vB_t1", "vB_t2"]:
        if data_part[0:2] == "vA":
            print("  -", data_part + ": use scaler_vA")
            use_this_scaler = scaler_vA
        else:
            print("  -", data_part + ": use scaler_vB")
            use_this_scaler = scaler_vB
        trainSplit_scaled[data_part] = use_this_scaler.transform(trainSplit_raw[data_part].to_numpy())
        validSplit_scaled[data_part] = use_this_scaler.transform(validSplit_raw[data_part].to_numpy())
        testSplit_scaled[data_part] = use_this_scaler.transform(testSplit_raw[data_part].to_numpy())

        trainSplit_scaled["mask_" + data_part] = trainSplit_raw["mask_" + data_part]
        validSplit_scaled["mask_" + data_part] = validSplit_raw["mask_" + data_part]
        testSplit_scaled["mask_" + data_part] = testSplit_raw["mask_" + data_part]

    return {
        'scaler_viewA': scaler_vA,
        'scaler_viewB': scaler_vB,
        'trainSplit_scaled': trainSplit_scaled,
        'validSplit_scaled': validSplit_scaled,
        'testSplit_scaled': testSplit_scaled
    }


class OmicsDataset(Dataset):
    def __init__(self, dataset_scaled):

        self.data_viewA_time1 = dataset_scaled['vA_t1']
        self.data_viewB_time1 = dataset_scaled['vB_t1']
        self.data_viewA_time2 = dataset_scaled['vA_t2']
        self.data_viewB_time2 = dataset_scaled['vB_t2']

        self.mask_viewA_time1 = dataset_scaled['mask_vA_t1'].to_numpy()
        self.mask_viewB_time1 = dataset_scaled['mask_vB_t1'].to_numpy()
        self.mask_viewA_time2 = dataset_scaled['mask_vA_t2'].to_numpy()
        self.mask_viewB_time2 = dataset_scaled['mask_vB_t2'].to_numpy()

        self.observation_map = dataset_scaled['observation_map']

        assert len(self.data_viewA_time1) == len(self.data_viewB_time1), \
            'data_viewA_time1 and data_viewA_time2 should have the same length!'

        assert len(self.data_viewA_time1) == len(self.data_viewA_time2), \
            'data_viewA_time1 and data_viewA_time2 should have the same length!'

        if self.data_viewB_time2 is not None:
            assert len(self.data_viewA_time1) == len(self.data_viewB_time2), \
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

        mask_viewA_time1 = from_numpy(self.mask_viewA_time1[idx, :].astype(np.float32))
        mask_viewB_time1 = from_numpy(self.mask_viewB_time1[idx, :].astype(np.float32))
        mask_viewA_time2 = from_numpy(self.mask_viewA_time2[idx, :].astype(np.float32))
        mask_viewB_time2 = from_numpy(self.mask_viewB_time2[idx, :].astype(np.float32))

        observation_map = from_numpy(self.observation_map[idx, :].astype(np.float32))

        data = {'batch_viewA_time1': observe_viewA_time1,
                'batch_viewB_time1': observe_viewB_time1,
                'batch_viewA_time2': observe_viewA_time2,
                'batch_viewB_time2': observe_viewB_time2,

                'mask_viewA_time1': mask_viewA_time1,
                'mask_viewB_time1': mask_viewB_time1,
                'mask_viewA_time2': mask_viewA_time2,
                'mask_viewB_time2': mask_viewB_time2,

                'observation_map': observation_map}

        return data


def prepare_dataset(data_dir="data/MGH_COVID", valSet_ratio=0.2, trainNum="all",
                    obsNum=0, use_scaler="standard", save_data_dir=None, set_seed=1):
    dataSplit_raw = load_split_data(data_dir=data_dir, valSet_ratio=valSet_ratio,
                                    trainNum=trainNum, obsNum=obsNum,
                                    save_data_dir=save_data_dir, set_seed=set_seed)
    dataSplit_scaled = scale_data(dataSplit_raw_dict=dataSplit_raw, use_scaler=use_scaler)

    train_set = OmicsDataset(dataSplit_scaled['trainSplit_scaled'])
    val_set = OmicsDataset(dataSplit_scaled['validSplit_scaled'])
    test_set = OmicsDataset(dataSplit_scaled['testSplit_scaled'])

    output_dataset = {
        'train_set': train_set,
        'val_set': val_set,
        'test_set': test_set,
        'scaler_viewA': dataSplit_scaled['scaler_viewA'],
        'scaler_viewB': dataSplit_scaled['scaler_viewB'],
        'dataSplit_raw': dataSplit_raw
    }

    return output_dataset
