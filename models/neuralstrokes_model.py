"""
Copyright (C) 2021 Adobe. All rights reserved.
"""

from .neuralstrokesbase_model import NeuralStrokesBaseModel


class NeuralStrokesModel(NeuralStrokesBaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser = NeuralStrokesBaseModel.modify_commandline_options(parser, is_train)

        parser.add_argument('--lambda_identity', type=float, default=1.0, help='weight for L1 loss')

        parser.set_defaults(
                            beta1=0.0,
                            beta2=0.99,
                            )

        if is_train:
            parser.set_defaults(save_epoch_freq=1,
                                save_latest_freq=20000,
                                )
        else:
            parser.set_defaults(batch_size=1,
                                num_test=1,
                                )

        return parser

    def __init__(self, opt):
        super().__init__(opt)

    def compute_D_loss(self):
        if self.opt.lambda_GAN > 0.0:
            GAN_loss_D = super().compute_D_loss()
        else:
            GAN_loss_D = 0.0
        self.loss_D = GAN_loss_D
        return self.loss_D

    def compute_G_loss(self):
        loss_G = super().compute_G_loss()
        return loss_G


