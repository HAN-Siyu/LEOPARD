import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from itertools import chain
from itertools import permutations
from src.model import LEOPARD
from src.utils import *


def compute_sim_mtx(similar_pair1_A, similar_pair1_B, similar_pair2_A, similar_pair2_B, center):
    if center:
        norm_similar_pair1_A = F.normalize(similar_pair1_A - similar_pair1_A.mean(dim=-1, keepdim=True), dim=-1)
        norm_similar_pair1_B = F.normalize(similar_pair1_B - similar_pair1_B.mean(dim=-1, keepdim=True), dim=-1)
        norm_similar_pair2_A = F.normalize(similar_pair2_A - similar_pair2_A.mean(dim=-1, keepdim=True), dim=-1)
        norm_similar_pair2_B = F.normalize(similar_pair2_B - similar_pair2_B.mean(dim=-1, keepdim=True), dim=-1)

    else:
        norm_similar_pair1_A = F.normalize(similar_pair1_A, dim=-1)
        norm_similar_pair1_B = F.normalize(similar_pair1_B, dim=-1)
        norm_similar_pair2_A = F.normalize(similar_pair2_A, dim=-1)
        norm_similar_pair2_B = F.normalize(similar_pair2_B, dim=-1)

    similar_mtx1 = torch.sum(norm_similar_pair1_A * norm_similar_pair1_B, dim=-1)
    similar_mtx2 = torch.sum(norm_similar_pair2_A * norm_similar_pair2_B, dim=-1)
    positive_mtx = torch.cat([similar_mtx1, similar_mtx1, similar_mtx2, similar_mtx2], dim=0)

    concat_mtx = torch.cat([norm_similar_pair1_A, norm_similar_pair1_B,
                            norm_similar_pair2_A, norm_similar_pair2_B], dim=0)
    full_sim_mtx = torch.mm(concat_mtx, concat_mtx.t().contiguous())

    mask = (torch.ones_like(full_sim_mtx) - torch.eye(4 * norm_similar_pair1_A.size()[0],
                                                      device=full_sim_mtx.device)).bool()
    negative_mtx = full_sim_mtx.masked_select(mask).view(4 * norm_similar_pair1_A.size()[0], -1)

    return positive_mtx, negative_mtx


def compute_sim_loss(positive_mtx, negative_mtx, temperature):
    positive_part = torch.exp(positive_mtx / temperature).nan_to_num()

    negative_part = torch.exp(negative_mtx / temperature).nan_to_num()
    negative_part = negative_part.sum(dim=-1).nan_to_num()

    loss = -torch.log(positive_part / negative_part).nan_to_num()

    return loss.mean()


class TrainLEOPARD(pl.LightningModule):
    def __init__(self, train_set, val_set,

                 scaler_viewA, scaler_viewB,

                 pre_layers_viewA, pre_layers_viewB,
                 post_layers_viewA, post_layers_viewB,

                 encoder_content_layers, encoder_content_norm, encoder_content_dropout,
                 encoder_temporal_layers, encoder_temporal_norm, encoder_temporal_dropout,

                 generator_block_num, generator_norm, generator_dropout, merge_mode,

                 discriminator_layers, discriminator_norm, discriminator_dropout,

                 reconstruction_loss, adversarial_loss,
                 weight_reconstruction, weight_adversarial, weight_representation,
                 weight_contrastive, contrastive_temperature,

                 batch_size, lr_G, lr_D, b1_G=0.9, b1_D=0.9,
                 lr_scheduler_G='LambdaLR', lr_scheduler_D='LambdaLR',

                 use_projection_head=False, projection_output_size=None,
                 test_set=None, note=''):
        super().__init__()

        self.save_hyperparameters('pre_layers_viewA', 'pre_layers_viewB',
                                  'post_layers_viewA', 'post_layers_viewB',

                                  'encoder_content_layers', 'encoder_content_norm', 'encoder_content_dropout',
                                  'encoder_temporal_layers', 'encoder_temporal_norm', 'encoder_temporal_dropout',

                                  'generator_block_num', 'generator_norm', 'generator_dropout', 'merge_mode',

                                  'discriminator_layers', 'discriminator_norm', 'discriminator_dropout',

                                  'reconstruction_loss', 'adversarial_loss',
                                  'weight_reconstruction', 'weight_adversarial', 'weight_representation',
                                  'weight_contrastive', 'contrastive_temperature',

                                  'lr_scheduler_G', 'lr_scheduler_D', 'lr_G', 'lr_D', 'b1_G', 'b1_D', 'batch_size',

                                  'use_projection_head', 'projection_output_size', 'note'
                                  )

        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.any_obs = any(train_set.observation_map)

        self.scaler_viewA = scaler_viewA
        self.scaler_viewB = scaler_viewB

        if reconstruction_loss == 'MAE':
            self.loss_F_reconstruction_noReduction = nn.L1Loss(reduction="none")
            self.loss_F_reconstruction = nn.L1Loss()
        elif reconstruction_loss == 'MSE':
            self.loss_F_reconstruction_noReduction = nn.MSELoss(reduction="none")
            self.loss_F_reconstruction = nn.MSELoss()
        else:
            raise Exception('reconstruct_loss only supports "MAE" or "MSE"!')

        if adversarial_loss == 'MSE':
            self.loss_F_adversarial = nn.MSELoss()
        elif adversarial_loss == 'BCE':
            self.loss_F_adversarial = nn.BCELoss()
        else:
            raise Exception('adversarial_loss only supports "MSE" or "BCE"!')

        viewA_data_size = train_set.data_viewA_time1.shape[1]
        viewB_data_size = train_set.data_viewB_time1.shape[1]

        discriminator_output_size = 4 if self.any_obs else 3

        self.leopard = LEOPARD(viewA_data_size=viewA_data_size, viewB_data_size=viewB_data_size,

                               pre_layers_viewA=pre_layers_viewA, pre_layers_viewB=pre_layers_viewB,

                               encoder_content_layers=encoder_content_layers,
                               encoder_content_norm=encoder_content_norm,
                               encoder_content_dropout=encoder_content_dropout,

                               encoder_temporal_layers=encoder_temporal_layers,
                               encoder_temporal_norm=encoder_temporal_norm,
                               encoder_temporal_dropout=encoder_temporal_dropout,

                               generator_block_num=generator_block_num, generator_norm=generator_norm,
                               generator_dropout=generator_dropout, merge_mode=merge_mode,

                               post_layers_viewA=post_layers_viewA, post_layers_viewB=post_layers_viewB,

                               discriminator_layers=discriminator_layers,
                               discriminator_norm=discriminator_norm,
                               discriminator_dropout=discriminator_dropout,
                               discriminator_output_size=discriminator_output_size,

                               use_projection_head=use_projection_head,
                               projection_output_size=projection_output_size)

    def create_lr_schedule(self, optim, lr_scheduler, offset=0):
        if lr_scheduler == "LambdaLR":
            out_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optim, lr_lambda=LambdaLR(n_epochs=self.trainer.max_epochs, offset=offset,
                                          decay_start_epoch=1).step)
        elif lr_scheduler == "SGDR":
            out_lr_scheduler = {'scheduler': torch.optim.lr_scheduler.CosineAnnealingLR(
                optim, T_max=self.trainer.max_epochs * len(self.train_dataloader()), eta_min=1e-16,
                last_epoch=-1), 'interval': 'step'}
        else:
            raise Exception('lr_scheduler should be one of "LambdaLR" or "SGDR" or None!')

        return out_lr_scheduler

    def configure_optimizers(self):

        params_G = [self.leopard.pre_encoder_viewA.parameters(), self.leopard.pre_encoder_viewB.parameters(),
                    self.leopard.encoder_content.parameters(), self.leopard.encoder_temporal.parameters(),
                    self.leopard.generator.parameters(),
                    self.leopard.post_generator_viewA.parameters(), self.leopard.post_generator_viewB.parameters()]

        params_D = [self.leopard.pre_encoder_viewA.parameters(), self.leopard.pre_encoder_viewB.parameters(),
                    self.leopard.discriminator.parameters()]

        if self.hparams.use_projection_head:
            params_G += [self.leopard.projector_content.parameters(), self.leopard.projector_temporal.parameters()]

        optim_G = torch.optim.Adam(chain(*params_G), lr=self.hparams.lr_G, betas=(self.hparams.b1_G, 0.999))
        optim_D = torch.optim.Adam(chain(*params_D), lr=self.hparams.lr_D, betas=(self.hparams.b1_D, 0.999))

        if self.hparams.lr_scheduler_G == "none" and self.hparams.lr_scheduler_D == "none":
            optimizers = [optim_G, optim_D]
            return optimizers
        else:
            lr_scheduler_G = self.create_lr_schedule(optim=optim_G, lr_scheduler=self.hparams.lr_scheduler_G)
            lr_scheduler_D = self.create_lr_schedule(optim=optim_D, lr_scheduler=self.hparams.lr_scheduler_D)
            return (
                {'optimizer': optim_G, 'lr_scheduler': lr_scheduler_G},
                {'optimizer': optim_D, 'lr_scheduler': lr_scheduler_D}
            )

    def forward(self, x, view, timePoint, process_content=True, process_temporal=True):
        # x: (tensor) input_data,
        # view: (str) "viewA" or "viewB"
        # timePoint: (str) "time1" or "time2"
        # process_content:  (bool) True or False
        # process_temporal: (bool) True or False

        encoded_content, encoded_temporal = self.leopard.forward(x=x, view=view, timePoint=timePoint,
                                                                 process_content=process_content,
                                                                 process_temporal=process_temporal)
        return encoded_content, encoded_temporal

    def transfer(self, encoded_content, encoded_temporal, view):
        # encoded_content:  (tensor) encoded_content
        # encoded_temporal: (tensor) encoded_temporal
        # view: (str) "viewA" or "viewB"

        output = self.leopard.transfer(encoded_content=encoded_content,
                                       encoded_temporal=encoded_temporal, view=view)
        return output

    def compute_loss_contrastive(self, similar_pair1_A, similar_pair1_B, similar_pair2_A, similar_pair2_B):
        positive_mtx, negative_mtx = compute_sim_mtx(similar_pair1_A, similar_pair1_B,
                                                     similar_pair2_A, similar_pair2_B,
                                                     center=True)
        loss = compute_sim_loss(positive_mtx=positive_mtx, negative_mtx=negative_mtx,
                                temperature=self.hparams.contrastive_temperature)
        return loss

    def compute_loss_adversarial(self, D_output_fake, view, timePoint, train_G_or_D,
                                 D_output_real=None, observation_map=None):
        if view == 'viewA':
            if timePoint == 'time1':
                col_idx = 0
            else:
                col_idx = 1
        else:
            if timePoint == 'time1':
                col_idx = 2
            else:
                col_idx = 3

        if col_idx != 3:
            D_output_fake_current_class = D_output_fake[:, col_idx:(col_idx + 1)]
        else:
            if any(observation_map):
                D_output_fake_current_class = D_output_fake[observation_map, col_idx:(col_idx + 1)]
            else:
                loss_real = loss_fake = 0
                return loss_real, loss_fake

        if train_G_or_D == 'G':
            label_ones = torch.ones(D_output_fake_current_class.shape[0], 1).type_as(D_output_fake_current_class)
            loss_fake = self.loss_F_adversarial(D_output_fake_current_class, label_ones)
            loss_real = None
        elif train_G_or_D == 'D':
            if D_output_real is None:
                raise Exception("data_raw cannot be None when training D!")

            if col_idx != 3:
                D_output_real_current_class = D_output_real[:, col_idx:(col_idx + 1)]
            else:
                D_output_real_current_class = D_output_real[observation_map, col_idx:(col_idx + 1)]

            label_ones = torch.ones(D_output_real_current_class.shape[0], 1).type_as(D_output_real_current_class)
            label_zeros = torch.zeros(D_output_fake_current_class.shape[0], 1).type_as(D_output_fake_current_class)

            loss_real = self.loss_F_adversarial(D_output_real_current_class, label_ones)
            loss_fake = self.loss_F_adversarial(D_output_fake_current_class, label_zeros)
        else:
            raise Exception("Wrong train_G_or_D!")

        return loss_real, loss_fake

    def complete_missing_view(self, batch):
        raw_viewA_time2 = batch['batch_viewA_time2']
        raw_viewB_time1 = batch['batch_viewB_time1']

        _, temporal_viewA_time2 = self.leopard.forward(x=raw_viewA_time2, view='viewA',
                                                       timePoint='time2')
        content_viewB_time1, _ = self.leopard.forward(x=raw_viewB_time1, view='viewB',
                                                      timePoint='time1')

        generated_viewB_time2 = self.leopard.transfer(encoded_content=content_viewB_time1,
                                                      encoded_temporal=temporal_viewA_time2,
                                                      view='viewB')
        return generated_viewB_time2

    def training_step(self, batch, batch_idx, optimizer_idx):
        raw_viewA_time1 = batch['batch_viewA_time1']
        raw_viewA_time2 = batch['batch_viewA_time2']
        raw_viewB_time1 = batch['batch_viewB_time1']

        mask_viewA_time1 = batch['mask_viewA_time1'].bool().detach()
        mask_viewA_time2 = batch['mask_viewA_time2'].bool().detach()
        mask_viewB_time1 = batch['mask_viewB_time1'].bool().detach()
        mask_viewB_time2 = batch['mask_viewB_time2'].bool().detach()

        content_viewA_time1, temporal_viewA_time1 = self.forward(x=raw_viewA_time1, view='viewA',
                                                                 timePoint='time1')
        content_viewA_time2, temporal_viewA_time2 = self.forward(x=raw_viewA_time2, view='viewA',
                                                                 timePoint='time2')
        content_viewB_time1, temporal_viewB_time1 = self.forward(x=raw_viewB_time1, view='viewB',
                                                                 timePoint='time1')

        generated_viewB_time2 = self.transfer(encoded_content=content_viewB_time1,
                                              encoded_temporal=temporal_viewA_time2,
                                              view='viewB')
        content_viewB_time2_gen, temporal_viewB_time2_gen = self.forward(x=generated_viewB_time2,
                                                                         view='viewB',
                                                                         timePoint='time2')

        content_viewB_time2_ref, temporal_viewB_time2_ref = content_viewB_time1.clone(), temporal_viewA_time2.clone()
        data_viewB_time2 = generated_viewB_time2.clone()

        observation_map = batch['observation_map'].bool().detach().cpu().numpy()[:, 0]
        # print(self.current_epoch, observation_map.sum())
        if any(observation_map):
            raw_viewB_time2 = batch['batch_viewB_time2']
            content_viewB_time2_obs, temporal_viewB_time2_obs = self.forward(x=raw_viewB_time2,
                                                                             view='viewB',
                                                                             timePoint='time2')
            content_viewB_time2_ref[observation_map, :] = content_viewB_time2_obs[observation_map, :]
            temporal_viewB_time2_ref[observation_map, :] = temporal_viewB_time2_obs[observation_map, :]

            content_viewB_time2_gen[observation_map, :] = content_viewB_time2_obs[observation_map, :]
            temporal_viewB_time2_gen[observation_map, :] = temporal_viewB_time2_obs[observation_map, :]

            data_viewB_time2[observation_map, :] = raw_viewB_time2[observation_map, :]

        # raw_library acts like a reference. Generated data will be compared with the data in raw_library.
        raw_library = {
            "viewA": {
                "time1": {"data": raw_viewA_time1, "mask": mask_viewA_time1,
                          "content": content_viewA_time1, "temporal": temporal_viewA_time1,
                          "view": "viewA", "timePoint": "time1"},

                "time2": {"data": raw_viewA_time2, "mask": mask_viewA_time2,
                          "content": content_viewA_time2, "temporal": temporal_viewA_time2,
                          "view": "viewA", "timePoint": "time2"}
            },
            "viewB": {
                "time1": {"data": raw_viewB_time1, "mask": mask_viewB_time1,
                          "content": content_viewB_time1, "temporal": temporal_viewB_time1,
                          "view": "viewB", "timePoint": "time1"},

                "time2": {"data": data_viewB_time2, "mask": mask_viewB_time2,
                          "content": content_viewB_time2_ref, "temporal": temporal_viewB_time2_ref,
                          "view": "viewB", "timePoint": "time2"}
            }
        }
        # Representations in content and temporal tools are used for data reconstruction/generation.
        content_pool = {
            "viewA_time1": {"data": content_viewA_time1, "timePoint": "time1", "view": "viewA"},
            "viewA_time2": {"data": content_viewA_time2, "timePoint": "time2", "view": "viewA"},
            "viewB_time1": {"data": content_viewB_time1, "timePoint": "time1", "view": "viewB"},
            "viewB_time2": {"data": content_viewB_time2_gen, "timePoint": "time2", "view": "viewB"}
        }
        temporal_pool = {
            "viewA_time1": {"data": temporal_viewA_time1, "timePoint": "time1", "view": "viewA"},
            "viewA_time2": {"data": temporal_viewA_time2, "timePoint": "time2", "view": "viewA"},
            "viewB_time1": {"data": temporal_viewB_time1, "timePoint": "time1", "view": "viewB"},
            "viewB_time2": {"data": temporal_viewB_time2_gen, "timePoint": "time2", "view": "viewB"}
        }

        representation_pairs = list(permutations(["viewA_time1", "viewA_time2",
                                                  "viewB_time1", "viewB_time2"], 2))

        ###### train G ######
        if optimizer_idx == 0:
            ###### contrastive loss ######
            if self.hparams.use_projection_head:
                loss_contrastive_content = self.hparams.weight_contrastive * self.compute_loss_contrastive(
                    similar_pair1_A=self.leopard.projector_content(content_viewA_time1),
                    similar_pair1_B=self.leopard.projector_content(content_viewA_time2),
                    similar_pair2_A=self.leopard.projector_content(content_viewB_time1),
                    similar_pair2_B=self.leopard.projector_content(content_viewB_time2_gen)
                )
                loss_contrastive_temporal = self.hparams.weight_contrastive * self.compute_loss_contrastive(
                    similar_pair1_A=self.leopard.projector_temporal(temporal_viewA_time1),
                    similar_pair1_B=self.leopard.projector_temporal(temporal_viewB_time1),
                    similar_pair2_A=self.leopard.projector_temporal(temporal_viewA_time2),
                    similar_pair2_B=self.leopard.projector_temporal(temporal_viewB_time2_gen)
                )
            else:
                loss_contrastive_content = self.hparams.weight_contrastive * self.compute_loss_contrastive(
                    similar_pair1_A=content_viewA_time1,
                    similar_pair1_B=content_viewA_time2,
                    similar_pair2_A=content_viewB_time1,
                    similar_pair2_B=content_viewB_time2_gen
                )
                loss_contrastive_temporal = self.hparams.weight_contrastive * self.compute_loss_contrastive(
                    similar_pair1_A=temporal_viewA_time1,
                    similar_pair1_B=temporal_viewB_time1,
                    similar_pair2_A=temporal_viewA_time2,
                    similar_pair2_B=temporal_viewB_time2_gen
                )

            loss_contrastive = (loss_contrastive_content + loss_contrastive_temporal) / 2

            ### arbitrary data generation using content and temporal representation pairs
            loss_reconstruction_sum = 0
            loss_representation_content_sum = 0
            loss_representation_temporal_sum = 0
            loss_adversarial_sum = 0

            for selected_pair in representation_pairs:
                content_label = selected_pair[0]
                temporal_label = selected_pair[1]

                current_content = content_pool[content_label]
                current_temporal = temporal_pool[temporal_label]
                current_target = raw_library[current_content['view']][current_temporal['timePoint']]

                generated_data = self.transfer(encoded_content=current_content['data'],
                                               encoded_temporal=current_temporal['data'],
                                               view=current_content['view'])
                content_generation, temporal_generation = self.forward(x=generated_data,
                                                                       view=current_content['view'],
                                                                       timePoint=current_temporal['timePoint'])

                ###### reconstruction loss ######
                loss_reconstruction_current = self.loss_F_reconstruction_noReduction(generated_data, current_target['data'])
                loss_reconstruction_mask = loss_reconstruction_current[current_target['mask']]
                loss_reconstruction_sum += torch.mean(loss_reconstruction_mask)

                ###### representation loss ######
                loss_representation_content_sum += self.loss_F_reconstruction(content_generation,
                                                                              current_target['content'])
                loss_representation_temporal_sum += self.loss_F_reconstruction(temporal_generation,
                                                                               current_target['temporal'])

                ######  adversarial  loss  ######
                _, D_output_generated = self.leopard.discriminator(
                    self.leopard.pre_encode(generated_data, view=current_content['view'],
                                            timePoint=current_temporal['timePoint'])
                )
                _, loss_adversarial_fake = self.compute_loss_adversarial(D_output_fake=D_output_generated,
                                                                         view=current_content['view'],
                                                                         timePoint=current_temporal['timePoint'],
                                                                         train_G_or_D='G', D_output_real=None,
                                                                         observation_map=observation_map)
                loss_adversarial_sum += loss_adversarial_fake

            loss_reconstruction = self.hparams.weight_reconstruction * loss_reconstruction_sum / len(representation_pairs)
            loss_adversarial = self.hparams.weight_adversarial * loss_adversarial_sum / ((len(representation_pairs) - 3) + (3 * any(observation_map)))
            loss_representation_content = self.hparams.weight_representation * loss_representation_content_sum / len(representation_pairs)
            loss_representation_temporal = self.hparams.weight_representation * loss_representation_temporal_sum / len(representation_pairs)
            loss_representation = (loss_representation_content + loss_representation_temporal) / 2

            loss = loss_contrastive + loss_reconstruction + loss_adversarial + loss_representation

            self.log("a0.train_G_contrastive", {'content': loss_contrastive_content,
                                                'temporal': loss_contrastive_temporal},
                     on_step=False, on_epoch=True, logger=True)

            self.log("a1.train_G_representation", {'content': loss_representation_content,
                                                   'temporal': loss_representation_temporal},
                     on_step=False, on_epoch=True, logger=True)

            self.log("a2.train_G_overall", {'contrastive': loss_contrastive,
                                            'reconstruction': loss_reconstruction,
                                            'adversarial': loss_adversarial,
                                            'representation': loss_representation},
                     on_step=False, on_epoch=True, logger=True)
            return {'loss': loss}
        ###### end train G ######

        ###### train D ######
        if optimizer_idx == 1:
            loss_adversarial_sum_real = 0
            loss_adversarial_sum_fake = 0

            for selected_pair in representation_pairs:
                content_label = selected_pair[0]
                temporal_label = selected_pair[1]

                current_content = content_pool[content_label]
                current_temporal = temporal_pool[temporal_label]
                current_target = raw_library[current_content['view']][current_temporal['timePoint']]

                generated_data = self.transfer(encoded_content=current_content['data'],
                                               encoded_temporal=current_temporal['data'],
                                               view=current_content['view'])

                ######  adversarial  loss  ######
                _, D_output_generated = self.leopard.discriminator(
                    self.leopard.pre_encode(generated_data, view=current_content['view'],
                                            timePoint=current_temporal['timePoint'])
                )
                _, D_output_observed = self.leopard.discriminator(
                    self.leopard.pre_encode(current_target['data'], view=current_content['view'],
                                            timePoint=current_temporal['timePoint'])
                )

                loss_adversarial_real, loss_adversarial_fake = self.compute_loss_adversarial(
                    D_output_fake=D_output_generated,
                    D_output_real=D_output_observed,
                    view=current_content['view'],
                    timePoint=current_temporal['timePoint'],
                    train_G_or_D='D', observation_map=observation_map)
                loss_adversarial_sum_real += loss_adversarial_real
                loss_adversarial_sum_fake += loss_adversarial_fake

            loss_adversarial_real = loss_adversarial_sum_real / ((len(representation_pairs) - 3) + (3 * any(observation_map)))
            loss_adversarial_fake = loss_adversarial_sum_fake / ((len(representation_pairs) - 3) + (3 * any(observation_map)))
            loss = (loss_adversarial_real + loss_adversarial_fake) / 2

            self.log("a3.train_D_adversarial", {'real (observed)': loss_adversarial_real,
                                                'fake (generated)': loss_adversarial_fake},
                     on_step=False, on_epoch=True, logger=True)
            return {'loss': loss}
        ###### end train D ######

    def validation_step(self, batch, batch_idx, log=True):
        metric_PB = self.predict_step(batch, batch_idx)

        self.log("b.validation_PB", metric_PB['quantile_percentBias'],
                 on_step=False, on_epoch=True, logger=True)
        return metric_PB

    def test_step(self, batch, batch_idx):
        metric_PB = self.predict_step(batch, batch_idx)
        self.log("c.test_PB", metric_PB['quantile_percentBias'],
                 on_step=False, on_epoch=True, logger=True)
        return metric_PB

    def predict_step(self, batch, batch_idx):
        generated_data = self.complete_missing_view(batch)
        observed_data = batch['batch_viewB_time2']

        metric_PB = compute_percent_bias(generated_data_numpy=generated_data.detach().cpu().numpy(),
                                         observed_data_numpy=observed_data.detach().cpu().numpy(),
                                         scaler=self.scaler_viewB)
        return metric_PB

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_set, shuffle=True,
                                           batch_size=self.hparams.batch_size,
                                           num_workers=0, pin_memory=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_set, shuffle=False,
                                           batch_size=self.val_set.data_viewA_time1.shape[0],
                                           num_workers=0, pin_memory=True)

    def test_dataloader(self):
        if self.test_set is not None:
            return torch.utils.data.DataLoader(self.test_set, shuffle=False,
                                               batch_size=self.test_set.data_viewA_time1.shape[0],
                                               num_workers=0, pin_memory=True)
        else:
            return None
