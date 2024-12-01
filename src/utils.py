import os
import numpy as np
import pandas as pd
import torch.nn as nn


class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)


class ExceptionObsNum(ValueError):  # raised if data conversion fails
    def __init__(self, varLabel, zeroBound, a, b):
        print('The ', varLabel, ' should be ', zeroBound, ' and <= the number of available training samples.\n',
              'Your ', varLabel, ' is ', a,
              '.\nThe number of available training samples in this split is ',
              b, "!", sep="")


def compute_percent_bias(generated_data_numpy, observed_data_numpy, scaler):
    generated_recover = scaler.inverse_transform(generated_data_numpy)
    observed_recover = scaler.inverse_transform(observed_data_numpy)

    percentBias = np.abs(generated_recover - observed_recover) / (np.abs(observed_recover) + 1e-16)
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
