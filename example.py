import torch
import pytorch_lightning as pl
import pandas as pd
from pytorch_lightning.loggers import TensorBoardLogger

from src.data import *
from src.train import TrainLEOPARD

for obsNum in [0, 25, 50, 100]:
    loaded_data = prepare_dataset(data_dir="data/MGH_COVID", valSet_ratio=0.2,
                                  obsNum=obsNum, set_seed=1, use_scaler="standard")

    pl.seed_everything(seed=1, workers=True)
    my_leopard = TrainLEOPARD(train_set=loaded_data['train_set'], val_set=loaded_data['val_set'],
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
                              lr_scheduler_G='none', lr_scheduler_D='none',

                              use_projection_head=False, projection_output_size=0,
                              batch_size=64, note="obsNum_" + str(obsNum))

    save_dir = "lightning_logs"
    name = "experiments"
    trainer = pl.Trainer(
        enable_progress_bar=True,
        log_every_n_steps=3,
        max_epochs=100,
        gpus=1 if torch.cuda.is_available() else None,
        logger=TensorBoardLogger(save_dir=save_dir, name=name)
    )

    trainer.fit(my_leopard)

    imputed_data = trainer.predict(my_leopard, my_leopard.test_dataloader())[0]

    output_dir = os.path.join(save_dir, name, "version_" + str(trainer.logger.version), "results")
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    pd.DataFrame(imputed_data["generated_data"]).to_csv(
        ## save in data/MGH_COVID_imputed folder:
        os.path.join("data/MGH_COVID_imputed", "obsNum_" + str(obsNum), "LEOPARD.csv"), index=False

        ## or save in the corresponding logger folder:
        # os.path.join(output_dir, "imputedData_obs" + str(obsNum) + ".csv"), index=False
    )
    # compute PB if groundtruth is available:
    # pd.DataFrame(imputed_data["raw_percentBias"]).to_csv(
    #             os.path.join(output_dir, "PB_obs" + str(obsNum) + ".csv"), index=False
    # )
