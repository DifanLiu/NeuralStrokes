"""
Copyright (C) 2021 Adobe. All rights reserved.
"""

import numpy as np
import os.path
from data.base_dataset import BaseDataset
from util.utilNS import collect_con_feat
from NSutil import ifn2img_tensor_full, ifn2img_tensor_full_gs, tensor_resize_crop_cat, tensor_resize_crop_cat_diffvg, collect_rtsc_feats
import torch
import torch.nn.functional as F


class STTrainDataset(BaseDataset):  # training set class for Stroke Texture
    def __init__(self, opt):
        BaseDataset.__init__(self, opt)

        self.crop_levels_B = np.array([64, 128, 192, 256]).astype(np.int16)  # 4 different scales

        self.B_pm = {}

        self.B_plain = 1.0 - ifn2img_tensor_full_gs(os.path.join(opt.dataroot, 'artist_drawing_mask_diffvg.png'))  # 0-background

        if opt.NS_have_color:
            self.B_stylized = 1.0 - ifn2img_tensor_full(os.path.join(opt.dataroot, 'artist_drawing.png')) # Ground Truth
        else:
            self.B_stylized = 1.0 - ifn2img_tensor_full_gs(os.path.join(opt.dataroot, 'artist_drawing.png'))


        self.B_pm['plain'] = self.B_plain

        self.B_pm = collect_rtsc_feats(self.B_pm, os.path.join(opt.dataroot, 'features'))

        self.B_condition = collect_con_feat(self.B_pm, opt.NS_con_feat)  # 3D geometric features

        self.dataset_size = self.opt.NS_atom_iterations * self.opt.batch_size

        temp_scale_ids = np.random.choice(4, size=self.opt.NS_atom_iterations)

        # make sure that each batch has the same crop size
        self.crops_id_np = np.reshape(np.tile(temp_scale_ids[:, np.newaxis], (1, self.opt.batch_size)), [-1])

        self.B_crop_dir = os.path.join(opt.dataroot, 'pool')  # the POOL of effective crop positions


    def __getitem__(self, index):

        self.B_in = self.B_pm['plain']

        # get crop scale from self.crops_id_np
        scale_id = self.crops_id_np[index]
        train_crop_size = self.crop_levels_B[scale_id]
        train_batch_size = 1

        B_crop_np = np.load(os.path.join(self.B_crop_dir, str(scale_id) + '.npy'))

        crop_random_id = np.random.randint(B_crop_np.shape[0])  # randomly select the crop position
        B_crop_np_small = B_crop_np[crop_random_id, :]
        B_crop_np_small = B_crop_np_small[np.newaxis, :]  # 1, 2

        aug_degree = np.random.uniform() * 360.0  # rotation augmentation

        A, A_con, A_crop_pm_list = tensor_resize_crop_cat_diffvg(self.B_pm, self.B_in,
                                                                 self.B_condition,
                                                                 1.0, 1.0,
                                                                 B_crop_np_small, train_batch_size,
                                                                 crop_size=train_crop_size,
                                                                 no_diff=self.opt.NS_no_diff,
                                                                 aug_degree=aug_degree)
        B, _ = tensor_resize_crop_cat(self.B_stylized, self.B_condition,
                                          1.0, 1.0,
                                          B_crop_np_small, train_batch_size, crop_size=train_crop_size,
                                          aug_degree=aug_degree)

        # ---- random real crops for discriminator
        crop_random_id_2 = np.random.randint(B_crop_np.shape[0])
        B_crop_np_small_2 = B_crop_np[crop_random_id_2, :]
        B_crop_np_small_2 = B_crop_np_small_2[np.newaxis, :]  # 1, 2

        C, _ = tensor_resize_crop_cat(self.B_stylized, self.B_condition,
                                      1.0, 1.0,
                                      B_crop_np_small_2, train_batch_size, crop_size=train_crop_size,
                                      aug_degree=np.random.uniform() * 360.0)

        # ---- get mask tensor for GAN loss computation
        weight_tensor = torch.zeros(1, 1, 70, 70, dtype=torch.float32)
        center_size = self.opt.NS_center_size  # effective receptive field of basic discriminator
        start_idx = int((70 - center_size) / 2)
        weight_tensor[0, 0, start_idx:(start_idx + center_size), start_idx:(start_idx + center_size)] = 1.0

        if A.shape[1] == 1:  # grayscale image
            temp_tensor = F.conv2d(A, weight_tensor, stride=8, padding=23)  # 1, 1, s, s
        else:  # colorful image
            temp_tensor = F.conv2d(torch.max(A, dim=1, keepdim=True)[0], weight_tensor, stride=8, padding=23)  # 1, 1, s, s

        temp_np = temp_tensor.cpu().numpy()
        temp_np[temp_np > 0.0] = 1.0  # binary mask: 1 means the area contains strokes, 0 means no strokes
        temp_np = temp_np.astype(np.float32)
        fake_mask_tensor = torch.from_numpy(temp_np)

        if C.shape[1] == 1:
            temp_tensor_real = F.conv2d(C, weight_tensor, stride=8, padding=23)  # 1, 1, s, s
        else:
            temp_tensor_real = F.conv2d(torch.max(C, dim=1, keepdim=True)[0], weight_tensor, stride=8, padding=23)  # 1, 1, s, s

        temp_np_real = temp_tensor_real.cpu().numpy()
        temp_np_real[temp_np_real > 0.0] = 1.0
        temp_np_real = temp_np_real.astype(np.float32)
        real_mask_tensor = torch.from_numpy(temp_np_real)

        real_mask_tensor = real_mask_tensor[0, :, :, :]  # 1, s, s
        fake_mask_tensor = fake_mask_tensor[0, :, :, :]

        A = A[0, :, :, :]
        A_con = A_con[0, :, :, :]
        B = B[0, :, :, :]

        ans_dict = {'A': 1.0 - A, 'B': 1.0 - B, 'A_con': 1.0 - A_con, 'train_crop_size': train_crop_size}
        ans_dict['real_mask'] = real_mask_tensor
        ans_dict['fake_mask'] = fake_mask_tensor

        C = C[0, :, :, :]
        ans_dict['C'] = 1.0 - C


        return ans_dict # 1-background


    def __len__(self):
        """ the single image contains multiple training instances.
        """
        return self.dataset_size


