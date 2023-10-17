# LEOPARD

**Missing view comp<ins>l</ins>etion for multi-tim<ins>e</ins>point <ins>o</ins>mics data via re<ins>p</ins>resentation disent<ins>a</ins>nglement and tempo<ins>r</ins>al knowle<ins>d</ins>ge transfer**

Longitudinal multi-view omics data offer unique insights into the temporal dynamics of individual-level physiology, which provides opportunities to advance personalized healthcare. However, the common occurrence of incomplete views makes extrapolation tasks difficult, and there is a lack of tailored methods for this critical issue. Here, we introduce LEOPARD, an innovative approach specifically designed to complete missing views in multi-timepoint omics data. By disentangling longitudinal omics data into content and temporal representations, LEOPARD transfers temporal knowledge to omics-specific content, thereby completing missing views. Compared to conventional imputation methods, LEOPARD yields the most robust results across three benchmark datasets. LEOPARD-imputed data also achieve the best agreement with the observed data in our analyses for age-associated metabolites detection and chronic kidney disease prediction. Our work takes the first step toward a generalized treatment of missing views in longitudinal omics data, enabling comprehensive exploration of temporal dynamics and providing valuable insights into personalized healthcare.

*Thank you for checking our LEOPARD!*

*Any questions regarding LEOPARD please drop an email to siyu.han@tum.de or post it to [issues](https://github.com/HAN-Siyu/LEOPARD/issues).*


## Habitat
Specific environment settings are required to run LEOPARD.

- python: 3.79
- cuda: 11.3
- pytorch: 1.11.10
- pytorch_lightning: 1.6.4
- tensorboard: 2.10.0

## How to Train Your LEOPARD
The architecture of LEOPARD is fully customizable and supports data of two views. LEOPARD is better to run in an interactive mode. Examples are provided in `main.py`.

```python                       
your_leopard = TrainLEOPARD(train_set, val_set, est_set,
                            scaler_viewA, scaler_viewB,

                            pre_layers_viewA, pre_layers_viewB,
                            post_layers_viewA, post_layers_viewB,
    
                            encoder_content_layers,
                            encoder_content_norm,
                            encoder_content_dropout,
    
                            encoder_temporal_layers,
                            encoder_temporal_norm,
                            encoder_temporal_dropout,
    
                            generator_block_num,
                            generator_norm,
                            generator_dropout,
                            merge_mode,
    
                            discriminator_layers,
                            discriminator_norm,
                            discriminator_dropout,
    
                            reconstruction_loss, adversarial_loss,
                            weight_reconstruction, weight_adversarial,
                            weight_representation,
                            weight_contrastive,
                            contrastive_temperature,
    
                            lr_G, lr_D, b1_G, b1_D,
                            lr_scheduler_G, lr_scheduler_D,
    
                            use_projection_head, projection_output_size,
                            batch_size, note)

trainer = pl.Trainer(
    enable_progress_bar=True,
    log_every_n_steps=5,
    max_epochs=100,
    gpus=1 if torch.cuda.is_available() else None,
    logger=TensorBoardLogger("lightning_logs/experiments", name="")
)

trainer.fit(your_leopard)
```

- `train_set`, `val_set`, `test_set`: data used for training, validation and test
- `scaler_viewA`, `scaler_viewB`: scaler used to transform data in two views
- `pre_layers_viewA`, `pre_layers_viewB`: pre-layers for view A and view B to convert them to the same dimension. Default: `[64]`, `[64]`
- `post_layers_viewA`, `post_layers_viewB`: post-layers for view A and view B to convert embeddings back to data in original dimension. Default: `[64]`, `[64]`

- `encoder_content_layers`: layers for the content encoder. A list where the length indicates the total number of layers, and each element specifies the size of the corresponding layer. Default: `[64, 64, 64]`
- `encoder_content_norm`: a list indicates if using normalization for the layers in the content encoder. Supported `"instance"`, `"batch"`, and `"none"`. Default: `['instance', 'instance', 'instance']`
- `encoder_content_dropout`: a list specifies dropout rate for each layer in the content encoder. Default: `[0, 0, 0]`
    
- `encoder_temporal_layers`: layers for the temporal encoder. Default: `[64, 64, 64]`
- `encoder_temporal_norm`: if use normalization for the layers in the temporal encoder? Supported `"instance"`, `"batch"`, and `"none"`. Default: `['none', 'none', 'none']`
- `encoder_temporal_dropout`: dropout rate for each layer in the temporal encoder. Default: `[0, 0, 0]`
    
- `generator_block_num`: how many layers/blocks used for the generator. Default: `3`
- `generator_norm`: if use normalization for the layers in the generator? Supported `"instance"`, `"batch"`, and `"none"`. Default: `['none', 'none', 'none']`
- `generator_dropout`: dropout rate for each layer in the generator. Default: `[0, 0, 0]`
- `merge_mode`: re-entangle content and temporal representations by concatenation (`"concat"`) or AdaIN (`"adain"`)? Default: `"adain"`
    
- `discriminator_layers`: layers for the multi-task discriminator. Default: `[128, 128]`
- `discriminator_norm`: if use normalization for the layers in the discriminator? Supported `"instance"`, `"batch"`, and `"none"`. Default: `['none', 'none']`
- `discriminator_dropout`: dropout rate for each layer in the discriminator. Default: `[0, 0]`
    
- `reconstruction_loss`: use `"MSE"` or `"MAE"` to compute reconstruction loss? Default: `"MSE"`
- `adversarial_loss`: use `"MSE"` or `"BCE"` to compute adversarial loss? Default: `"MSE"`
- `weight_reconstruction`, `weight_adversarial`, `weight_representation`, `weight_contrastive`: weights for different losses. Default: `1`, `1`, `0.1`, `0.1`
- `contrastive_temperature`: temperature for NT-Xent-based contrastive loss. Default: `0.05`
    
- `lr_G`, `lr_D`: learning rate for generator process (encoders and generator) and discrimination process (discriminator). You need to tune this for your own datasets. Default: `0.005`, `0.05`
- `b1_G`, `b1_D`: beta_1 for Adam Optimizer. Default: `0.9`, `0.9`
- `lr_scheduler_G`, `lr_scheduler_D`: `"none"` or use `"LambdaLR"` or `"SGDR"` as lr scheduler? Default: `none`, `none`
    
- `use_projection_head`: if use projection head for contrastive learning? Default: `False`
- `projection_output_size`: set output size of projection head. Ignored if `use_projection_head=False`.  Default: `0`
- `batch_size`: batch size. Default: `64`
- `note`: add some additional texts as a hyperparameter to label each run.  Default: `""`

## Script Files

- main.py: examples for running LEOPARD
- src/data.py: dataset preparation
- src/layers.py: basic layers used to build LEOPARD
- src/model.py: class of LEOPARD architecture
- src/train.py: LightningModule of LEOPARD training
- src/utils.py: some utility functions for data processing.

## Cite This Work

If you use the code or data in this repository, please cite:

```bibtex
@article{han2023missing,
  title={Missing view completion for multi-timepoints omics data via representation disentanglement and temporal knowledge transfer},
  author={Han, S and Yu, S and Shi, M and Harada, M and Ge, J and Lin, J and Prehn, C and Petrera, A and Li, Y and Sam, F and others},
  journal={biorxiv preprint doi:10.1101/2023.09.26.559302},
  year={2023}
}
```

## Our Big Cat Zoo

- [TIGER](https://github.com/HAN-Siyu/TIGER): technical variation elimination for metabolomics data using ensemble learning architecture
- [LION](https://github.com/HAN-Siyu/LION): an integrated R package for effective prediction of lncRNA/ncRNA–protein interaction
- LEOPARD (this work): missing view completion for multi-timepoint omics data via representation disentanglement and temporal knowledge transfer