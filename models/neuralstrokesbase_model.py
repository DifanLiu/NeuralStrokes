"""
Copyright (C) 2021 Adobe. All rights reserved.
"""

import torch
from .base_model import BaseModel
from . import networks
import pydiffvg
from util.utilNS import get_svg_shapes, remove_alpha, get_pm_attributes_from_1D


class NeuralStrokesBaseModel(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """  Configures options specific for NeuralStrokes model
        """

        parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss')

        parser.set_defaults(pool_size=0)

        opt, _ = parser.parse_known_args()

        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)

        # -----initialization used for pydiffvg
        pydiffvg.set_print_timing(False)
        pydiffvg.set_use_gpu(True)
        self.render = pydiffvg.RenderFunction.apply
        # -----

        # specify the training losses you want to print out.
        # The training/test scripts will call <BaseModel.get_current_losses>
        if self.opt.lambda_GAN > 0.0:
            self.loss_names = ['G_GAN', 'D_real', 'D_fake', 'G']
            self.visual_names = ['real_A', 'fake_B', 'real_B']
            self.visual_names.append('real_C')
            self.visual_names.append('real_mask_tensor')
        else:
            self.loss_names = ['G']
            self.visual_names = ['real_A', 'fake_B', 'real_B']

        if self.opt.lambda_smoothing > 0.0:
            self.visual_names.append('smooth_fake')
            self.visual_names.append('smooth_styled')

        if self.isTrain and self.opt.lambda_GAN > 0.0:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']

        # define networks (generator)
        self.netG = networks.define_G(opt.NS_ic + opt.NS_cc, opt.NS_oac, opt.ngf, opt.netG, opt.normG, not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias, opt.no_antialias_up, self.gpu_ids, opt)

        if self.isTrain:
            self.criterionIdt = torch.nn.L1Loss().to(self.device)  # L1 loss
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizers.append(self.optimizer_G)

            if self.opt.lambda_GAN > 0.0:
                self.netD = networks.define_D(opt.NS_oc, opt.ndf, opt.netD, opt.n_layers_D, opt.normD,
                                              opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
                self.criterionGAN = networks.GANLoss(opt.gan_mode, use_mask=True).to(self.device)
                self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
                self.optimizers.append(self.optimizer_D)
                if self.opt.lambda_smoothing > 0.0:
                    self.smoothing = networks.GaussianSmoothing(opt.NS_oc, 10, 10).to(self.device)  # reference --> SketchPatch: Sketch Stylization via Seamless Patch-level Synthesis

    def data_dependent_initialize(self, data):
        self.set_input(data)
        self.forward()                     # compute fake images
        if self.opt.isTrain:
            if self.opt.lambda_GAN > 0.0:
                self.compute_D_loss().backward()                  # calculate gradients for D
            self.compute_G_loss().backward()                   # calculate graidents for G

    def optimize_parameters(self):
        # forward
        self.forward()

        # update D
        if self.opt.lambda_GAN > 0.0:
            self.set_requires_grad(self.netD, True)
            self.optimizer_D.zero_grad()
            self.loss_D = self.compute_D_loss()
            self.loss_D.backward()
            self.optimizer_D.step()
            self.set_requires_grad(self.netD, False)
        # update G
        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()


    def set_input(self, input):
        if self.opt.NS_no_diff:  # Stroke Texture
            self.real_A = input['A'].to(self.device)
            self.A_con = input['A_con'].to(self.device)
            self.real_B = input['B'].to(self.device)

            self.canvas_hw = input['train_crop_size'].cpu().numpy()[0]
            # ----
            self.A_crop_pm_list = []
            self.image_paths = ''
            self.real_mask_tensor = input['real_mask'].to(self.device)  # bs, 1, s, s
            self.fake_mask_tensor = input['fake_mask'].to(self.device)
            self.real_C = input['C'].to(self.device)

        else:  # Stroke Geometry
            self.real_A = input['A'].to(self.device)
            self.A_con = input['A_con'].to(self.device)
            self.real_B = input['B'].to(self.device)

            self.canvas_hw = input['train_crop_size']
            # ----
            the_list = ['ans_list', 'indices_list', 'norm_list', 'tangent_list', 'ODfeat_list']
            self.A_crop_pm_list = []
            for adict in input['A_crop_pm_list']:  # loop over batch_id
                new_dict = {}
                for key, value in adict.items():
                    if key not in the_list:
                        continue
                    new_value = []
                    for astroke in value:
                        new_value.append(astroke.to(self.device))
                    new_dict[key] = new_value
                self.A_crop_pm_list.append(new_dict)

    def set_input_test(self, input):
        self.real_A = input['A'].to(self.device)
        self.A_con = input['A_con'].to(self.device)

        # ----
        the_list = ['ans_list', 'indices_list', 'norm_list', 'tangent_list', 'ODfeat_list']
        self.A_crop_pm_list = []
        for adict in input['A_crop_pm_list']:  # loop over batch_id
            new_dict = {}
            for key, value in adict.items():
                if key not in the_list:
                    assert 0
                new_value = []
                for astroke in value:
                    new_value.append(astroke.to(self.device))
                new_dict[key] = new_value
            self.A_crop_pm_list.append(new_dict)
        self.canvas_hw = self.real_A.shape[2]

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        use_1DConv = not self.opt.NS_no_diff
        if self.opt.NS_1d_con_feat != '':
            assert use_1DConv

        self.real = torch.cat((self.real_A, self.A_con), 1)
        if use_1DConv:  # SG
            self.attributes_features, self.attributes_list = self.netG(self.real, self.A_crop_pm_list, use_flip=self.opt.NS_OD_flip)
            fake_B_list = []
            self.dpxy_batch_list = []
            for bid in range(self.attributes_features.shape[0]):  # loop over batch
                dp_list, th_list = get_pm_attributes_from_1D(self.A_crop_pm_list[bid],
                                                             self.attributes_list[bid],
                                                             min_thickness=self.opt.NS_min_th)
                scene_args = get_svg_shapes(self.A_crop_pm_list[bid], th_list, hw=self.canvas_hw, dpxy_list=dp_list)

                if self.opt.lambda_TV_dp > 0.0:
                    self.dpxy_batch_list.append(dp_list)

                pred_img = remove_alpha(self.render(self.canvas_hw, self.canvas_hw, 2, 2, 1, None, *scene_args))  # differentiable rendering
                fake_B_list.append(pred_img.unsqueeze(0))  # 1, h, w

            self.fake_B = torch.stack(fake_B_list, 0)
        else:  # ST
            self.fake_B = self.netG(self.real)

    def compute_D_loss(self):
        """Calculate GAN loss for the unconditional discriminator"""

        fake = self.fake_B.detach()

        # Fake; stop backprop to the generator by detaching fake_B

        pred_fake = self.netD(fake)

        self.loss_D_fake = self.criterionGAN(pred_fake, False, self.fake_mask_tensor).mean()

        # Real

        real_temp = self.real_C

        self.pred_real = self.netD(real_temp)

        loss_D_real = self.criterionGAN(self.pred_real, True, self.real_mask_tensor)

        self.loss_D_real = loss_D_real.mean()

        # combine loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D

    def compute_G_loss(self):
        """Calculate loss for the generator"""
        fake = self.fake_B
        # First, G(A) should fool the discriminator
        if self.opt.lambda_GAN > 0.0:
            fake_temp = fake
            pred_fake = self.netD(fake_temp)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True, self.fake_mask_tensor).mean() * self.opt.lambda_GAN
        else:
            self.loss_G_GAN = 0.0

        loss_NS_paired = torch.nn.functional.l1_loss(self.fake_B, self.real_B) * self.opt.lambda_identity  # L1 loss
        self.loss_G = self.loss_G_GAN + loss_NS_paired

        if self.opt.lambda_smoothing > 0.0:  # reference --> SketchPatch: Sketch Stylization via Seamless Patch-level Synthesis
            self.smooth_fake = self.smoothing(self.fake_B)
            self.smooth_styled = self.smoothing(self.real_B)
            loss_gaussian_smoothing = torch.nn.functional.l1_loss(self.smooth_fake, self.smooth_styled) * self.opt.lambda_smoothing
            self.loss_G = self.loss_G + loss_gaussian_smoothing

        if self.opt.lambda_TV_dp > 0.0:  # displacement regularization
            loss_TV_dp = 0.0  # sum at the beginning
            total_points_TV = 0.0
            assert len(self.dpxy_batch_list) > 0  # not empty
            for bid in range(len(self.dpxy_batch_list)):
                the_pred_DP_list = self.dpxy_batch_list[bid]  # a list of curves
                for cid in range(len(the_pred_DP_list)):
                    small_pred_DP = the_pred_DP_list[cid].permute(1, 0)  # nv, 2
                    if small_pred_DP.shape[0] >= 2:
                        delta_neighbors = small_pred_DP[0:-1, :] - small_pred_DP[1:, :]  # nv, 2
                        loss_TV_dp += torch.sum(delta_neighbors ** 2)
                        total_points_TV += delta_neighbors.shape[0]
            self.loss_G = self.loss_G + loss_TV_dp * self.opt.lambda_TV_dp / total_points_TV

        return self.loss_G
