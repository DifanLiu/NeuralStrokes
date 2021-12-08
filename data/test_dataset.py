"""
Copyright (C) 2021 Adobe. All rights reserved.
"""

import numpy as np
import os.path
from data.base_dataset import BaseDataset
from util.utilNS import ss2vg, collect_con_feat
from NSutil import ifn2img_tensor_full_gs, tensor_resize_crop_cat_diffvg, collect_rtsc_feats


class TestDataset(BaseDataset):  # testing set class for Neural Strokes
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        assert opt.phase == 'test'

        self.pm_list = []
        self.condition_list = []

        this_data_dir = opt.dataroot
        this_pm = ss2vg(os.path.join(this_data_dir, 'pm_ES.pkl'))  # planar map
        this_plain = 1.0 - ifn2img_tensor_full_gs(os.path.join(this_data_dir, 'plain.png'))  # 0-background
        this_pm['plain'] = this_plain
        this_pm = collect_rtsc_feats(this_pm, os.path.join(this_data_dir, 'features'))
        this_condition = collect_con_feat(this_pm, opt.NS_con_feat)  # 3D geometric features
        self.pm_list.append(this_pm)
        self.condition_list.append(this_condition)


    def __getitem__(self, index):

        the_pm = self.pm_list[index]
        the_in = the_pm['plain']
        the_condition = self.condition_list[index]

        rid = 0
        cid = 0

        A, A_con, A_crop_pm_list = tensor_resize_crop_cat_diffvg(the_pm, the_in, the_condition,
                                                                 1.0, 1.0,
                                                                 np.array([[rid, cid]], dtype=np.int16),
                                                                 self.opt.batch_size,
                                                                 crop_size=self.opt.crop_size,
                                                                 no_diff=self.opt.NS_no_diff,
                                                                 con_feat_1d=self.opt.NS_1d_con_feat)


        return {'A': 1.0 - A, 'A_con': 1.0 - A_con,
                'A_crop_pm_list': A_crop_pm_list}  # 1-background


    def __len__(self):  # one instance
        return len(self.pm_list)
