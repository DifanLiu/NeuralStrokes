"""
Copyright (C) 2021 Adobe. All rights reserved.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import scipy.ndimage as ndimage
import glob
import os


def ifn2mask_tensor(ifn, h_scale, w_scale):
    base_np = cv2.imread(ifn, cv2.IMREAD_GRAYSCALE)
    ori_h = base_np.shape[0]
    ori_w = base_np.shape[1]
    new_h = int(round(h_scale * ori_h))
    new_w = int(round(w_scale * ori_w))

    img = 1.0 - cv2.resize(base_np, (new_w, new_h), interpolation=cv2.INTER_AREA) / 255.0
    img = img.astype(np.float32)
    ans_tensor = torch.from_numpy(img).cuda()
    return ans_tensor.unsqueeze(0).unsqueeze(0)


def ifn2img_tensor(ifn, h_scale, w_scale, rid, cid, crop_size=64):
    base_np = cv2.imread(ifn, cv2.IMREAD_COLOR)

    ori_h = base_np.shape[0]
    ori_w = base_np.shape[1]
    new_h = int(round(h_scale * ori_h))
    new_w = int(round(w_scale * ori_w))

    img = cv2.resize(base_np, (new_w, new_h), interpolation=cv2.INTER_AREA) / 255.0
    img = img.astype(np.float32)
    ans_tensor = torch.from_numpy(img)
    ans_tensor = ans_tensor[rid:(rid + crop_size), cid:(cid + crop_size), :]  # 64, 64, 3
    return ans_tensor.permute(2, 0, 1)  # 3, 64, 64


def ifn2img_tensor_full(ifn):
    img = cv2.imread(ifn, cv2.IMREAD_COLOR) / 255.0
    img = img.astype(np.float32)
    ans_tensor = torch.from_numpy(img)
    return ans_tensor.permute(2, 0, 1)  # 3, h, w


def ifn2img_tensor_full_gs(ifn):

    img = cv2.imread(ifn, cv2.IMREAD_GRAYSCALE) / 255.0
    img = img.astype(np.float32)
    ans_tensor = torch.from_numpy(img)
    return ans_tensor.unsqueeze(0)  # 1, h, w


def collect_rtsc_feats(adict, feat_dir):
    png_list = glob.glob(feat_dir + '/*.png')
    ss_list = [os.path.split(temp)[1].split('.')[0] for temp in png_list]

    for ss_idx, ss in enumerate(ss_list):
        ifn = png_list[ss_idx]
        new_dict_key = ss.split('_')[0]
        max_value = float(ss.split('_')[1])
        img = cv2.imread(ifn, cv2.IMREAD_UNCHANGED) / max_value
        assert img.ndim == 2
        img = img.astype(np.float32)
        ans_tensor = torch.from_numpy(img)
        adict[new_dict_key] = 1.0 - ans_tensor.unsqueeze(0)

    return adict


def conv2d(mask_tensor, weight):
    return F.conv2d(mask_tensor, weight)


def tensor_resize_crop_cat(input_tensor, condition_tensor, h_scale, w_scale, crops_np, bs, crop_size=64,
                           aug_degree=0.0):  # all 0 background

    input_list = []
    condition_list = []

    ori_h = input_tensor.shape[1]
    ori_w = input_tensor.shape[2]
    new_h = int(round(h_scale * ori_h))
    new_w = int(round(w_scale * ori_w))
    if h_scale == 1.0 and w_scale == 1.0:
        input_resize = input_tensor.unsqueeze(0)
        condition_resize = condition_tensor.unsqueeze(0)
    else:
        input_resize = F.interpolate(input_tensor.unsqueeze(0), size=(new_h, new_w), mode='nearest')  # 1, 1, nh, nw
        condition_resize = F.interpolate(condition_tensor.unsqueeze(0), size=(new_h, new_w), mode='nearest')

    for bid in range(bs):
        rid = crops_np[bid, 0]
        cid = crops_np[bid, 1]
        input_list.append(rotate_tensor(input_resize[:, :, rid:(rid + crop_size), cid:(cid + crop_size)], aug_degree, inter_type='LINEAR'))  # 1, 1, h, w

        condition_list.append(rotate_tensor(condition_resize[:, :, rid:(rid + crop_size), cid:(cid + crop_size)], aug_degree))

    final_input_tensor = torch.cat(input_list, 0)  # bs, C, h, w
    final_condition_tensor = torch.cat(condition_list, 0)

    return final_input_tensor, final_condition_tensor


def rotate_tensor(input_tensor, aug_degree=0.0, inter_type='NEAREST'):  # 1, C, h, w  rotate tensor with NN
    if aug_degree == 0.0:  # no rotation
        return input_tensor
    else:
        ans_list = []

        for c_dim in range(input_tensor.shape[1]):
            input_np = np.uint16(input_tensor[0, c_dim, :, :].numpy() * 65535.0)  # h, w  use cv2 to do the rotation and interpolation
            h, w = input_np.shape
            M = cv2.getRotationMatrix2D((w / 2, h / 2), aug_degree, 1)  # counterclockwise

            the_border_value = 0

            if inter_type == 'NEAREST':
                dst = cv2.warpAffine(input_np, M, (w, h), flags=cv2.INTER_NEAREST, borderValue=the_border_value)  # cv2.INTER_NEAREST
            elif inter_type == 'LINEAR':
                dst = cv2.warpAffine(input_np, M, (w, h), flags=cv2.INTER_LINEAR, borderValue=the_border_value)  # cv2.INTER_NEAREST
            else:
                assert 0

            dst_float = dst / 65535.0
            dst_float = dst_float.astype(np.float32)
            ans_list.append(dst_float[np.newaxis, :, :])  # 1, h, w
        return torch.from_numpy(np.concatenate(ans_list, axis=0)).unsqueeze(0)


def tensor_resize_crop_cat_diffvg(pm, input_tensor, condition_tensor, h_scale, w_scale, crops_np, bs, crop_size=64,
                                  no_diff=False, aug_degree=0.0,
                                  con_feat_1d=''):
    # ----
    if no_diff:  # ST
        assert con_feat_1d == ''
    else:  # SG
        assert con_feat_1d != ''
        assert aug_degree == 0.0

    input_list = []  # a list of images
    condition_list = []
    pm_list = []
    ori_h = input_tensor.shape[1]
    ori_w = input_tensor.shape[2]
    new_h = int(round(h_scale * ori_h))
    new_w = int(round(w_scale * ori_w))
    if not no_diff:
        new_pm_ans_padded = rescale_pm_ans_padded(pm['ans_padded'], new_h, new_w, h=ori_h * 1.0, w=ori_w * 1.0)  # num_strokes, max_v, 2  scaled

    input_resize = F.interpolate(input_tensor.unsqueeze(0), size=(new_h, new_w), mode='nearest')  # 1, 1, nh, nw
    condition_resize = F.interpolate(condition_tensor.unsqueeze(0), size=(new_h, new_w), mode='nearest')  # 1, N, nh, nw

    for bid in range(bs):
        rid = crops_np[bid, 0]
        cid = crops_np[bid, 1]
        input_list.append(rotate_tensor(input_resize[:, :, rid:(rid + crop_size), cid:(cid + crop_size)], aug_degree, inter_type='LINEAR'))
        condition_list.append(rotate_tensor(condition_resize[:, :, rid:(rid + crop_size), cid:(cid + crop_size)], aug_degree))
        # crop pm
        if not no_diff:
            crop_pm = crop_pm_ans_padded(new_pm_ans_padded.clone(), pm['num_vertices_list'], rid, cid, new_h, new_w, crop_size,
                                         pm_dict=pm, con_feat_1d=con_feat_1d)  # a dict
            pm_list.append(crop_pm)

    final_input_tensor = torch.cat(input_list, 0)  # bs, C, h, w
    final_condition_tensor = torch.cat(condition_list, 0)

    return final_input_tensor, final_condition_tensor, pm_list


def crop_pm_ans_padded(a_padded, num_vertices_list, rid, cid, h, w, crop_size,
                       pm_dict=None, con_feat_1d=''):  # one crop and one pm
    # a_padded  vertices after rotation; NS, NV, 2

    r_bool = (a_padded[:, :, 1] >= rid) & (a_padded[:, :, 1] <= rid + crop_size - 1)  # torch.bool
    c_bool = (a_padded[:, :, 0] >= cid) & (a_padded[:, :, 0] <= cid + crop_size - 1)
    final_bool = r_bool & c_bool
    final_bool_np = final_bool.numpy()  # n_path, n_v   true means inside the patch

    norm_list = []
    tangent_list = []


    ans_list = []  # store strokes
    indices_list = []  # store indices
    ODfeat_list = []
    for path_id in range(final_bool_np.shape[0]):
        real_bool_single = final_bool_np[path_id, 0:num_vertices_list[path_id]]
        if np.sum(real_bool_single) > 1:  # 2, ... vertices  # this stroke will be used in the crop
            labeled_array, num_features = ndimage.label(real_bool_single * 1)  # 1, 2, num_features
            assert num_features >= 1
            for object_id in range(1, num_features + 1):  # from 1 to num_features   curves may be divided into multiple parts
                v_indices = np.nonzero(labeled_array == object_id)[0]  # nv
                if v_indices.shape[0] >= 2:  # at least two vertices
                    start_idx = np.amin(v_indices)
                    end_idx = np.amax(v_indices)  # inclusive
                    ans_list.append(a_padded[path_id, start_idx:(end_idx + 1), :] - torch.tensor([[cid, rid]], dtype=torch.float32))  # shifted to the crop nv, 2
                    pos_np_raw = ans_list[-1].numpy()  # nv, 2  # nv >= 2
                    pos_int = np.flip(np.around(pos_np_raw).astype(np.int64), axis=1)
                    pos_int[pos_int < 0] = 0
                    pos_int[pos_int > crop_size - 1] = crop_size - 1
                    indices_tensor = torch.from_numpy(pos_int.copy())  # lookup table  new indices list
                    indices_list.append(indices_tensor)

                    norm_list.append(pm_dict['norm_list'][path_id][start_idx:(end_idx + 1), :])  # nv, 2
                    tangent_list.append(pm_dict['tangent_list'][path_id][start_idx:(end_idx + 1), :])  # nv, 2

                    if con_feat_1d != '':  # collect 1D features
                        ODfeat_list.append(collect_ODfeat(pm_dict, path_id, start_idx, end_idx, con_feat_1d))

    final_ans_dict = {'ans_list': ans_list, 'indices_list': indices_list}

    final_ans_dict['norm_list'] = norm_list
    final_ans_dict['tangent_list'] = tangent_list

    if con_feat_1d != '':
        final_ans_dict['ODfeat_list'] = ODfeat_list
    return final_ans_dict


def collect_ODfeat(pm_dict, path_id, start_idx, end_idx, con_feat_1d):
    ans_list = []
    str_list = con_feat_1d.split('_')
    for astr in str_list:
        ans_list.append(pm_dict[astr][path_id][start_idx:(end_idx + 1), :])
    return torch.cat(ans_list, dim=1)  # N, C


def rescale_pm_ans_padded(a_padded, new_h, new_w, h=768, w=768):

    return a_padded * torch.tensor([[[new_w / w, new_h / h]]], dtype=torch.float32)


def rescale_pm_ans_list(a_list, new_h, new_w, h=768, w=768):
    ans_list = []
    for a_stroke in a_list:
        ans_list.append(a_stroke * torch.tensor([[new_w / w, new_h / h]], dtype=torch.float32))
    return ans_list
