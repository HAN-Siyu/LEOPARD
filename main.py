import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger

from src.utils import *
from src.train import TrainLEOPARD

loaded_data = prepare_dataset(root_dir="data", benchmark_data="MGH_COVID", valSet_ratio=0.2,
                              obsNum=0, set_seed=1, use_scaler="standard")

pl.seed_everything(1, workers=True)

self = TrainLEOPARD(train_set=loaded_data['train_set'], val_set=loaded_data['val_set'],
                    test_set=loaded_data['test_set'],
                    scaler_viewA=loaded_data['scaler_viewA'], scaler_viewB=loaded_data['scaler_viewB'],

                    pre_layers_viewA=[64], pre_layers_viewB=[64],
                    post_layers_viewA=[64], post_layers_viewB=[64],

                    encoder_content_layers=[64, 64, 64],
                    encoder_content_norm=['instance', 'instance', 'instance'],
                    encoder_content_dropout=[0, 0, 0],

                    encoder_temporal_layers=[64, 64, 64],
                    encoder_temporal_norm=['none', 'none', 'none'],
                    encoder_temporal_dropout=[0, 0, 0],

                    generator_block_num=3,
                    generator_norm=['none', 'none', 'none'],
                    generator_dropout=[0, 0, 0],
                    merge_mode='adain',

                    discriminator_layers=[64, 64],
                    discriminator_norm=['none', 'none'],
                    discriminator_dropout=[0, 0],

                    reconstruction_loss='MSE', adversarial_loss='MSE',
                    weight_reconstruction=1, weight_adversarial=1,
                    weight_representation=0.1,
                    weight_contrastive=0.1,
                    contrastive_temperature=0.05,

                    lr_G=0.005, lr_D=0.05, b1_G=0.9, b1_D=0.9,
                    lr_scheduler_G='none', lr_scheduler_D='none',  # LambdaLR, none

                    use_projection_head=False, projection_output_size=0,
                    batch_size=64, note="")

trainer = pl.Trainer(
    enable_progress_bar=True,
    log_every_n_steps=5,
    max_epochs=100,
    gpus=1 if torch.cuda.is_available() else None,
    logger=TensorBoardLogger("lightning_logs/experiments", name="experiments")
)

trainer.fit(self)

