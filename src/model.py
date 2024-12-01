from src.layers import *


class LEOPARD(nn.Module):
    def __init__(self, viewA_data_size, viewB_data_size,

                 pre_layers_viewA, pre_layers_viewB,
                 encoder_content_layers, encoder_content_norm, encoder_content_dropout,
                 encoder_temporal_layers, encoder_temporal_norm, encoder_temporal_dropout,

                 generator_block_num, generator_norm, generator_dropout, merge_mode,
                 post_layers_viewA, post_layers_viewB,

                 discriminator_layers, discriminator_norm, discriminator_dropout, discriminator_output_size,

                 use_projection_head, projection_output_size):
        super().__init__()

        assert pre_layers_viewA[-1] == pre_layers_viewB[-1], \
            "the last layers of pre_layers_viewA and pre_layers_viewB should have the same size!"

        self.use_projection_head = use_projection_head

        # input dimension for pre-encoders: dimension of view size + 1
        # an additional dimension is used to label timePoint of the input data
        self.pre_encoder_viewA = Encoder(encoder_input_size=viewA_data_size + 1,
                                         encoder_layers_size=pre_layers_viewA,
                                         encoder_norm=['none'] * len(pre_layers_viewA),
                                         encoder_dropout=[0] * len(pre_layers_viewA))

        self.pre_encoder_viewB = Encoder(encoder_input_size=viewB_data_size + 1,
                                         encoder_layers_size=pre_layers_viewB,
                                         encoder_norm=['none'] * len(pre_layers_viewB),
                                         encoder_dropout=[0] * len(pre_layers_viewB))

        # input dimension for encoders: dimension of pre-layer's output + 1
        # an additional dimension is used to label the view and timePoint of the pre-encoded embeddings
        encoder_input_size = pre_layers_viewA[-1] + 2

        self.encoder_content = Encoder(encoder_input_size=encoder_input_size,
                                       encoder_layers_size=encoder_content_layers,
                                       encoder_norm=encoder_content_norm,
                                       encoder_dropout=encoder_content_dropout)

        self.encoder_temporal = Encoder(encoder_input_size=encoder_input_size,
                                        encoder_layers_size=encoder_temporal_layers,
                                        encoder_norm=encoder_temporal_norm,
                                        encoder_dropout=encoder_temporal_dropout)

        if merge_mode not in ["adain", "concat"]:
            raise Exception('merge_mode should "adain" or "concat"!')

        # If use "concat" to merge content and temporal representations,
        # generator_input_size needs to be multiplied by 2
        generator_input_size = (1 + (merge_mode != 'adain')) * encoder_content_layers[-1]

        self.generator = DecoderAdaIN(decoder_input_size=generator_input_size,
                                      decoder_block_num=generator_block_num,
                                      decoder_norm=generator_norm,
                                      decoder_dropout=generator_dropout,
                                      merge_mode=merge_mode)

        post_layers_viewA = post_layers_viewA + [viewA_data_size]  # add final output layer to post-layers
        self.post_generator_viewA = Encoder(encoder_input_size=generator_input_size,
                                            encoder_layers_size=post_layers_viewA,
                                            encoder_norm=['none'] * len(post_layers_viewA),
                                            encoder_dropout=[0] * len(post_layers_viewA))

        post_layers_viewB = post_layers_viewB + [viewB_data_size]  # add final output layer to post-layers
        self.post_generator_viewB = Encoder(encoder_input_size=generator_input_size,
                                            encoder_layers_size=post_layers_viewB,
                                            encoder_norm=['none'] * len(post_layers_viewB),
                                            encoder_dropout=[0] * len(post_layers_viewB))

        if use_projection_head:
            self.projector_content = ProjectionHead(input_size=encoder_content_layers[-1],
                                                    projection_output=projection_output_size)
            self.projector_temporal = ProjectionHead(input_size=encoder_temporal_layers[-1],
                                                     projection_output=projection_output_size)

        self.discriminator = Discriminator(input_size=encoder_input_size,
                                           output_size=discriminator_output_size,
                                           norm=discriminator_norm,
                                           layers_size=discriminator_layers,
                                           dropout=discriminator_dropout)

    def pre_encode(self, x, view, timePoint):
        # x: (tensor) input_data,
        # view: (str) "viewA" or "viewB"
        # timePoint: (str) "time1" or "time2"

        label_ones = torch.ones(x.shape[0], 1).type_as(x)
        label_zeros = torch.zeros(x.shape[0], 1).type_as(x)

        if timePoint == "time1":
            input_x = torch.cat((label_zeros, x), 1)
        elif timePoint == "time2":
            input_x = torch.cat((label_ones, x), 1)
        else:
            raise Exception("Wrong timePoint!")

        if view == "viewA":
            pre_encoded_data = self.pre_encoder_viewA(input_x)
        elif view == "viewB":
            pre_encoded_data = self.pre_encoder_viewB(input_x)
        else:
            raise Exception('Wrong view!')

        if timePoint == "time1":
            pre_encoded_data_addTime = torch.cat((label_zeros, pre_encoded_data), 1)
        else:
            pre_encoded_data_addTime = torch.cat((label_ones,  pre_encoded_data), 1)

        if view == "viewA":
            pre_encoded_data_addLabel = torch.cat((label_zeros, pre_encoded_data_addTime), 1)
        else:
            pre_encoded_data_addLabel = torch.cat((label_ones,  pre_encoded_data_addTime), 1)

        return pre_encoded_data_addLabel

    def forward(self, x, view, timePoint, process_content=True, process_temporal=True):
        # x: (tensor) input_data,
        # view: (str) "viewA" or "viewB"
        # timePoint: (str) "time1" or "time2"
        # process_content:  (bool) True or False
        # process_temporal: (bool) True or False

        # label_ones = torch.ones(x.shape[0], 1).type_as(x)
        # label_zeros = torch.zeros(x.shape[0], 1).type_as(x)

        pre_encoded_data_addLabel = self.pre_encode(x=x, view=view, timePoint=timePoint)

        if process_content:
            encoded_content = self.encoder_content(pre_encoded_data_addLabel)
        else:
            encoded_content = None

        if process_temporal:
            encoded_temporal = self.encoder_temporal(pre_encoded_data_addLabel)
        else:
            encoded_temporal = None

        return encoded_content, encoded_temporal

    def transfer(self, encoded_content, encoded_temporal, view):
        # encoded_content:  (tensor) encoded_content
        # encoded_temporal: (tensor) encoded_temporal
        # view: (str) "viewA" or "viewB"

        generator_embeddings = self.generator(encoded_content=encoded_content, encoded_temporal=encoded_temporal)

        if view == "viewA":
            out = self.post_generator_viewA(generator_embeddings)
        elif view == "viewB":
            out = self.post_generator_viewB(generator_embeddings)
        else:
            raise Exception("Wrong view!")

        return out
