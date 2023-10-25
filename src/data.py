from torch import from_numpy
from torch.utils.data import Dataset
from random import seed
from random import sample
from src.utils import *
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler


def load_split_data(data_dir, valSet_ratio, obsNum, set_seed):
    trainSplit_raw = dict()
    validSplit_raw = dict()
    testSplit_raw  = dict()

    valid_idx = None
    seed(set_seed)
    print("+ loading data:")
    for data_part in ["vA_t1", "vA_t2", "vB_t1", "vB_t2"]:
        print("  -", data_part)
        dataTrain = pd.read_csv(os.path.join(data_dir, data_part + "_train" + ".csv", ),
                                index_col=["label", "zz_nr"])

        dataTest  = pd.read_csv(os.path.join(data_dir, data_part + "_test" + ".csv", ),
                                index_col=["label", "zz_nr"])

        if valid_idx is None:
            valid_idx = sample(range(dataTrain.shape[0]), int(dataTrain.shape[0] * valSet_ratio))

        trainSplit_raw[data_part] = dataTrain.drop(dataTrain.index[valid_idx], axis=0, inplace=False)
        validSplit_raw[data_part] = dataTrain.iloc[valid_idx]
        testSplit_raw[data_part]  = dataTest

    trainSplit_raw["observation_map"] = np.zeros((trainSplit_raw["vB_t2"].shape[0], 1), dtype=bool)
    observed_idx_train = np.where(~trainSplit_raw["vB_t2"].iloc[:, 0].isnull().values)[0]
    if obsNum == "auto":
        trainSplit_raw["observation_map"][observed_idx_train] = True
    elif isinstance(obsNum, int):
        if 0 <= obsNum <= len(observed_idx_train):
            obsNum_idx = sample(observed_idx_train.tolist(), obsNum)
            trainSplit_raw["observation_map"][obsNum_idx] = True
        else:
            raise ExceptionObsNum(obsNum, len(observed_idx_train))
    else:
        raise ValueError("obsNum should be 'auto' or an integer!")

    validSplit_raw["observation_map"] = np.zeros((validSplit_raw["vB_t2"].shape[0], 1), dtype=bool)
    observed_idx_valid = np.where(~validSplit_raw["vB_t2"].iloc[:, 0].isnull().values)[0]
    validSplit_raw["observation_map"][observed_idx_valid] = True

    testSplit_raw["observation_map"] = np.zeros((testSplit_raw["vB_t2"].shape[0], 1), dtype=bool)
    observed_idx_test = np.where(~testSplit_raw["vB_t2"].iloc[:, 0].isnull().values)[0]
    testSplit_raw["observation_map"][observed_idx_test] = True

    return {
        'trainSplit_raw': trainSplit_raw,
        'validSplit_raw': validSplit_raw,
        'testSplit_raw':  testSplit_raw
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
    testSplit_scaled  = {"observation_map": testSplit_raw["observation_map"]}

    for data_part in ["vA_t1", "vA_t2", "vB_t1", "vB_t2"]:
        if data_part[0:2] == "vA":
            print("  -", data_part + ": use scaler_vA")
            use_this_scaler = scaler_vA
        else:
            print("  -", data_part + ": use scaler_vB")
            use_this_scaler = scaler_vB
        trainSplit_scaled[data_part] = use_this_scaler.transform(trainSplit_raw[data_part].to_numpy())
        validSplit_scaled[data_part] = use_this_scaler.transform(validSplit_raw[data_part].to_numpy())
        testSplit_scaled[data_part]  = use_this_scaler.transform(testSplit_raw[data_part].to_numpy())

    return {
        'scaler_viewA': scaler_vA,
        'scaler_viewB': scaler_vB,
        'trainSplit_scaled': trainSplit_scaled,
        'validSplit_scaled': validSplit_scaled,
        'testSplit_scaled':  testSplit_scaled
    }


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


def prepare_dataset(data_dir="data/MGH_COVID", valSet_ratio=0.2, obsNum=0, use_scaler="standard", set_seed=1):
    dataSplit_raw = load_split_data(data_dir=data_dir,
                                    valSet_ratio=valSet_ratio, obsNum=obsNum,
                                    set_seed=set_seed)
    dataSplit_scaled = scale_data(dataSplit_raw_dict=dataSplit_raw, use_scaler=use_scaler)

    train_set = OmicsDataset(dataSplit_scaled['trainSplit_scaled'])
    val_set   = OmicsDataset(dataSplit_scaled['validSplit_scaled'])
    test_set  = OmicsDataset(dataSplit_scaled['testSplit_scaled'])

    output_dataset = {
        'train_set': train_set,
        'val_set': val_set,
        'test_set': test_set,
        'scaler_viewA': dataSplit_scaled['scaler_viewA'],
        'scaler_viewB': dataSplit_scaled['scaler_viewB']
    }

    return output_dataset

