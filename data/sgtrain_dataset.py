"""
Copyright (C) 2021 Adobe. All rights reserved.
"""

import numpy as np
import os.path
from data.base_dataset import BaseDataset
from util.utilNS import ss2vg, collect_con_feat
from NSutil import ifn2img_tensor_full_gs, tensor_resize_crop_cat, tensor_resize_crop_cat_diffvg, collect_rtsc_feats


class SGTrainDataset(BaseDataset):  # training set class for Stroke Geometry
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.crop_levels_B = np.array([64, 128, 192, 256]).astype(np.int16)  # 4 different scales

        self.B_pm = ss2vg(os.path.join(opt.dataroot, 'pm_ES.pkl'))  # planar map

        self.B_plain = 1.0 - ifn2img_tensor_full_gs(os.path.join(opt.dataroot, 'plain.png'))  # 0-background

        self.B_stylized = 1.0 - ifn2img_tensor_full_gs(os.path.join(opt.dataroot, 'artist_drawing_mask.png'))  # Ground Truth

        self.B_pm['plain'] = self.B_plain

        self.B_pm = collect_rtsc_feats(self.B_pm, os.path.join(opt.dataroot, 'features'))

        self.B_condition = collect_con_feat(self.B_pm, opt.NS_con_feat)  # 3D geometric features

        self.B_crop_dir = os.path.join(opt.dataroot, 'pool')  # the POOL of effective crop positions

    def __getitem__(self, index):

        self.B_in = self.B_pm['plain']

        assert self.opt.phase == "train"

        # randomly select a crop scale
        num_sizes = self.crop_levels_B.shape[0]  # 4
        scale_id = np.random.choice(num_sizes)
        train_crop_size = self.crop_levels_B[scale_id]  # [64, 128, 192, 256]
        train_batch_size = self.opt.batch_size

        B_crop_np = np.load(os.path.join(self.B_crop_dir, str(scale_id) + '.npy'))

        np.random.shuffle(B_crop_np)

        A, A_con, A_crop_pm_list = tensor_resize_crop_cat_diffvg(self.B_pm, self.B_in,
                                                                 self.B_condition,
                                                                 1.0, 1.0,
                                                                 B_crop_np, train_batch_size,
                                                                 crop_size=train_crop_size,
                                                                 no_diff=self.opt.NS_no_diff,
                                                                 con_feat_1d=self.opt.NS_1d_con_feat)

        B, _ = tensor_resize_crop_cat(self.B_stylized, self.B_condition,
                                          1.0, 1.0,
                                          B_crop_np, train_batch_size, crop_size=train_crop_size)



        return {'A': 1.0 - A, 'B': 1.0 - B, 'A_con': 1.0 - A_con,
                'A_crop_pm_list': A_crop_pm_list, 'train_crop_size': train_crop_size}  # 1-background


    def __len__(self):
        """ the single image contains multiple training instances.
        """
        return self.opt.NS_atom_iterations

