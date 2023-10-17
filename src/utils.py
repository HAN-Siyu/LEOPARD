import os
import numpy as np
import pandas as pd
import torch.nn as nn
from random import seed
from random import sample
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
from src.data import OmicsDataset


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


class ExceptionObsNum(ValueError):  # raised if data conversion fails
    def __init__(self, a, b):
        print('The obsNum should be >= 0 and <= the number of observed samples in the training split.\n',
              'Your obsNum is ', a,
              '.\nThe number of observed samples in the training split is ',
              b, "!", sep="")


def compute_percent_bias(generated_data_numpy, observed_data_numpy, scaler):
    generated_recover = scaler.inverse_transform(generated_data_numpy)
    observed_recover = scaler.inverse_transform(observed_data_numpy)

    percentBias = np.abs((generated_recover - observed_recover) / (observed_recover + 1e-16))
    percentBias_median_var = np.median(percentBias, axis=0)
    percentBias_median = np.median(percentBias_median_var)
    percentBias_percentile_25 = np.percentile(percentBias_median_var, 25)
    percentBias_percentile_75 = np.percentile(percentBias_median_var, 75)

    return {'generated_data': generated_recover,
            'observed_data': observed_recover,
            'raw_percentBias': percentBias,
            'quantile_percentBias': {
                'per25': percentBias_percentile_25,
                'per50': percentBias_median,
                'per75': percentBias_percentile_75
            }
            }


def compute_cor_loss(pred, ref, dim):
    if dim == "col":
        cos_col = nn.CosineSimilarity(dim=0, eps=1e-8)
        pearson_col = 1 - cos_col(pred - pred.mean(dim=0, keepdim=True), ref - ref.mean(dim=0, keepdim=True))
        loss = pearson_col.mean()
    elif dim == "row":
        cos_row = nn.CosineSimilarity(dim=1, eps=1e-8)
        pearson_row = 1 - cos_row(pred - pred.mean(dim=1, keepdim=True), ref - ref.mean(dim=1, keepdim=True))
        loss = pearson_row.mean()
    elif dim == "both":
        cos_col = nn.CosineSimilarity(dim=0, eps=1e-8)
        pearson_col = 1 - cos_col(pred - pred.mean(dim=0, keepdim=True), ref - ref.mean(dim=0, keepdim=True))
        cos_row = nn.CosineSimilarity(dim=1, eps=1e-8)
        pearson_row = 1 - cos_row(pred - pred.mean(dim=1, keepdim=True), ref - ref.mean(dim=1, keepdim=True))
        loss = (pearson_col.mean() + pearson_row.mean()) / 2
    else:
        raise Exception('dim should be one of "col", "row", or "both"!')

    return loss


def load_split_data(root_dir, benchmark_data, valSet_ratio, obsNum, set_seed):
    trainSplit_raw = dict()
    validSplit_raw = dict()
    testSplit_raw  = dict()

    valid_idx = None
    seed(set_seed)

    for data_part in ["vA_t1", "vA_t2", "vB_t1", "vB_t2"]:
        print(data_part)
        dataTrain = pd.read_csv(os.path.join(root_dir, benchmark_data, data_part + "_train" + ".csv", ),
                                index_col=["label", "zz_nr"])

        dataTest  = pd.read_csv(os.path.join(root_dir, benchmark_data, data_part + "_test" + ".csv", ),
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
    print("- Data scaling using " + use_scaler + ".")

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
        print(data_part)
        if data_part[0:2] == "vA":
            print("use scaler_vA")
            use_this_scaler = scaler_vA
        else:
            print("use scaler_vB")
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


def prepare_dataset(root_dir, benchmark_data, valSet_ratio, obsNum, set_seed, use_scaler):
    dataSplit_raw = load_split_data(root_dir=root_dir,
                                    benchmark_data=benchmark_data,
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


def write_csv(save_path, output_metric_PB):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    pd.DataFrame(output_metric_PB['generated_data']).to_csv(
        os.path.join(save_path, "generated_data.csv"))
    pd.DataFrame(output_metric_PB['observed_data']).to_csv(
        os.path.join(save_path, "observed_data.csv"))
    pd.DataFrame(output_metric_PB['raw_percentBias']).to_csv(
        os.path.join(save_path, "raw_percentBias.csv"))
    print('percent bias:', output_metric_PB['quantile_percentBias'])
