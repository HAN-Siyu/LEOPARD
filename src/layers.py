import torch
import torch.nn as nn


def general_block(input_size, output_size, sigmoid=False, norm='instance', dropout=0.0, spectral_norm=False):
    if spectral_norm:
        layers = [nn.utils.parametrizations.spectral_norm(nn.Linear(input_size, output_size))]
    else:
        layers = [nn.Linear(input_size, output_size)]

    if norm == "batch":
        layers.append(nn.BatchNorm1d(num_features=output_size))
    elif norm == "instance":
        layers.append(nn.InstanceNorm1d(num_features=output_size))
    elif norm == "none":
        pass
    else:
        raise Exception('norm should be one of "none", "instance", or "batch"!')

    if sigmoid:
        layers.append(nn.Sigmoid())
    else:
        layers.append(nn.PReLU())

    if dropout:
        layers.append(nn.Dropout(p=dropout))
    return layers


class ProjectionHead(nn.Module):
    def __init__(self, input_size, projection_output):
        super().__init__()

        self.head = nn.Sequential(nn.Linear(input_size, input_size, bias=True),
                                  nn.ReLU(),
                                  nn.Linear(input_size, projection_output, bias=False))

    def forward(self, x):
        return self.head(x)


class AdaIN(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.instance_norm = nn.InstanceNorm1d(input_size, affine=False)

    def forward(self, encoded_content, encoded_temporal):
        content_normalized = self.instance_norm(encoded_content)

        temporal_mean = encoded_temporal.mean(1, keepdim=True)
        temporal_std = encoded_temporal.std(1, unbiased=True, keepdim=True)

        content_transformed = (content_normalized * temporal_std) + temporal_mean
        return content_transformed


class Block(nn.Module):
    def __init__(self, encoder_input_size, encoder_output_size,
                 norm='instance', dropout=0.0):
        super().__init__()
        self.block = nn.Sequential(*general_block(input_size=encoder_input_size,
                                                  output_size=encoder_output_size,
                                                  norm=norm, dropout=dropout))

    def forward(self, x):
        return self.block(x)


class Encoder(nn.Module):
    def __init__(self, encoder_input_size, encoder_layers_size, encoder_norm, encoder_dropout):
        super().__init__()

        encoder_list = []
        for i in range(len(encoder_layers_size)):
            if i != 0:
                encoder_input_size = encoder_layers_size[i - 1]

            encoder_list += [Block(encoder_input_size=encoder_input_size,
                                   encoder_output_size=encoder_layers_size[i],
                                   norm=encoder_norm[i], dropout=encoder_dropout[i])]

        self.encoder = nn.Sequential(*encoder_list)

    def forward(self, x):
        return self.encoder(x)


class DecoderAdaIN(nn.Module):
    def __init__(self, decoder_input_size, decoder_block_num, decoder_norm, decoder_dropout, merge_mode):
        super().__init__()

        self.merge_mode = merge_mode
        if self.merge_mode == "adain":
            self.AdaIN = AdaIN(decoder_input_size)

        self.decoder = nn.ModuleList()
        for i in range(decoder_block_num):
            self.decoder.append(Block(encoder_input_size=decoder_input_size,
                                      encoder_output_size=decoder_input_size,
                                      norm=decoder_norm[i], dropout=decoder_dropout[i]))

    def forward(self, encoded_content, encoded_temporal):
        if self.merge_mode == "concat":
            merged_embeddings = torch.cat((encoded_content, encoded_temporal), 1)

        for block in self.decoder:
            if self.merge_mode == "adain":
                merged_embeddings = self.AdaIN(encoded_content, encoded_temporal)
            decoded_data = block(merged_embeddings)

        return decoded_data


class Discriminator(nn.Module):
    def __init__(self, input_size, output_size, layers_size, norm, dropout):
        super().__init__()

        discriminator_list = []
        for i in range(len(layers_size)):
            if i != 0:
                input_size = layers_size[i - 1]

            discriminator_list += [*general_block(input_size=input_size,
                                                  output_size=layers_size[i],
                                                  norm=norm[i], dropout=dropout[i],
                                                  spectral_norm=False)]

        sigmoid_layer = general_block(input_size=layers_size[-1], output_size=output_size,
                                      sigmoid=True, norm="none", dropout=0.0)

        self.discriminator = nn.Sequential(*discriminator_list)
        self.sigmoid_layer = nn.Sequential(*sigmoid_layer)

    def forward(self, x):
        discriminator_embeddings = self.discriminator(x)
        discriminator_output = self.sigmoid_layer(discriminator_embeddings)
        return discriminator_embeddings, discriminator_output
