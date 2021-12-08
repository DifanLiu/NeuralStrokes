"""
Copyright (C) 2021 Adobe. All rights reserved.
"""

import cv2
import os
import numpy as np
import pickle
import torch
import pydiffvg
import torch.nn.functional as F


def txt2list(the_txt_fn):
    with open(the_txt_fn, 'r') as txt_obj:
        lines = txt_obj.readlines()
        lines = [haha.strip() for haha in lines]
    return lines


def list2txt(ofn, str_list):
    with open(ofn, 'a') as txt_obj:
        for small_str in str_list:
            txt_obj.write(small_str + '\n')


def show_img(cv2_array, ofn=None, title='image'):
    if ofn is None:
        cv2.imshow(title, cv2_array)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        cv2.imwrite(ofn, cv2_array)


def create_folder(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    return folder_name


def tensor2img(input_tensor):
    img = input_tensor.detach().cpu().numpy()
    img = np.clip(img, 0.0, 1.0) * 255.0
    img = np.uint8(img)
    return img


def tensor2npy(input_tensor):
    img = input_tensor.detach().cpu().numpy()
    return img


def mask_stroke_texture(SG_tensor, ST_tensor, iteration):
    mask = 1.0 - SG_tensor.squeeze().numpy()
    mask[mask > 0] = 1.0
    textured_drawing = ST_tensor[0, :, :, :].permute(1, 2, 0).cpu().numpy()

    assert iteration >= -1
    if iteration == -1:  # No masking
        pass
    elif iteration == 0:
        mask = mask[:, :, np.newaxis]
        textured_drawing = mask * textured_drawing + (1.0 - mask) * 1.0
    else:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(np.uint8(255.0 * mask), kernel, iterations=iteration) / 255.0
        mask = mask[:, :, np.newaxis]
        textured_drawing = mask * textured_drawing + (1.0 - mask) * 1.0
    if textured_drawing.shape[2] == 1:
        textured_drawing = textured_drawing[:, :, 0]

    return np.uint8(np.clip(textured_drawing, 0.0, 1.0) * 255.0)


def calculate_local_frame_polyline(vertices_np):  # nv, 2
    nv = vertices_np.shape[0]
    assert nv >= 2
    ans = np.zeros((nv, 2), dtype=np.float64)
    ans[0, :] = vertices_np[1, :] - vertices_np[0, :]
    ans[-1, :] = vertices_np[-1, :] - vertices_np[-2, :]
    if nv > 2:
        ans[1:-1, :] = vertices_np[2:, :] - vertices_np[:-2, :]
    l_array = np.sqrt(np.sum(ans ** 2, axis=1))  # tangent vector length
    assert np.sum(l_array == 0.0) == 0
    l_array = l_array[:, np.newaxis]
    ans = ans / l_array  # n, 2

    ans_normal = np.zeros((nv, 2), dtype=np.float64)
    ans_normal[:, 0] = ans[:, 1]
    ans_normal[:, 1] = -ans[:, 0]
    return ans, ans_normal


def ss2vg(list_fn, hw=768):  # read planar map
    ss = os.path.split(list_fn)[1].split('.')[0]
    with open(list_fn, 'rb') as f:
        input_list = pickle.load(f)  # a list of strokes

    num_strokes = len(input_list)
    assert num_strokes >= 1

    max_v = -1
    for a_idx, a_stroke in enumerate(input_list):
        assert len(a_stroke) >= 2
        if len(a_stroke) > max_v:
            max_v = len(a_stroke)

    ans_list = []  # a list of (variant, 2) tensor
    num_vertices_list = []  # a list of int   the mask  number of vertices
    features_np = np.zeros((num_strokes, 8, max_v), dtype=np.float32)  # zero padding  padded
    mask_np = np.zeros((num_strokes, max_v), dtype=np.float32)  # for loss computation
    norm_list = []
    tangent_list = []

    min_depth = 2e10 + 1.0
    max_depth = -2e10 + 1.0
    for a_idx, a_stroke in enumerate(input_list):
        temp = np.array(a_stroke).astype(np.float32)
        all_tensor = torch.from_numpy(temp)

        ans_list.append(all_tensor[:, 0:2].contiguous())
        NS_tangent, NS_normal = calculate_local_frame_polyline(temp[:, 0:2])  # n, 2; n, 2

        norm_list.append(torch.from_numpy(NS_normal.astype(np.float32)))
        tangent_list.append(torch.from_numpy(NS_tangent.astype(np.float32)))

        this_num_vertices = temp.shape[0]
        num_vertices_list.append(this_num_vertices)

        mask_np[a_idx, 0:this_num_vertices] = 1.0
        vertices_feature = np.arange(temp.shape[0]) / float(temp.shape[0])
        features_np[a_idx, 0, 0:this_num_vertices] = vertices_feature  # curve parameter 0
        features_np[a_idx, 1:3, 0:this_num_vertices] = temp[:, 0:2].T / hw  # x, y 1, 2
        features_np[a_idx, 3:7, 0:this_num_vertices] = temp[:, 2:6].T  # normal 3, 4, tangent 5, 6

        if np.amax(temp[:, -1]) > max_depth:
            max_depth = np.amax(temp[:, -1])
        if np.amin(temp[:, -1]) < min_depth:
            min_depth = np.amin(temp[:, -1])

    rasterized_pm = np.zeros((7, hw, hw), dtype=np.float32)  # zero background
    indices_list = []  # store num_path tensors
    for a_idx, a_stroke in enumerate(input_list):
        temp = np.array(a_stroke).astype(np.float32)[:, -1]  # depth np
        this_num_vertices = temp.shape[0]

        if max_depth == min_depth:  # no depth variation
            temp = temp * 0.0 + 1.0  # all 1.0
        else:  # normalize depth
            temp = (temp - min_depth) / (max_depth - min_depth)

        features_np[a_idx, -1, 0:this_num_vertices] = temp  # normalized depth, 7

        pos_int = np.flip(np.around(np.array(a_stroke)[:, 0:2].astype(np.float32)).astype(np.int64), axis=1)
        indices_tensor = torch.from_numpy(pos_int.copy())  # nv, 2
        indices_list.append(indices_tensor)

        rasterized_pm[0, pos_int[:, 0], pos_int[:, 1]] = 1.0  # mask channel
        rasterized_pm[1:6, pos_int[:, 0], pos_int[:, 1]] = features_np[a_idx, 0:5, 0:num_vertices_list[a_idx]]
        rasterized_pm[6, pos_int[:, 0], pos_int[:, 1]] = temp

    pm_tensor = torch.from_numpy(rasterized_pm)  # 7, hw, hw   :: dots_mask, arc, xy, normal, depth (white mask)

    features_tensor = torch.from_numpy(features_np)  # num_strokes, 8, max_v   ::arc, xy, normal, tangent, depth
    mask_tensor = torch.from_numpy(mask_np)  # num_strokes, max_v

    ans_padded = torch.zeros(num_strokes, max_v, 2, dtype=torch.float32)
    for path_id, a_tensor in enumerate(ans_list):
        ans_padded[path_id, :a_tensor.shape[0], :] = a_tensor


    ans_dict = {'ans_list': ans_list, 'norm_list': norm_list, 'tangent_list': tangent_list, 'num_vertices_list': num_vertices_list,
                'features_tensor': features_tensor, 'ss': ss, 'pm_tensor': pm_tensor, 'indices_list': indices_list,
                'mask_tensor': mask_tensor, 'mask': pm_tensor[0:1, :, :], 'depth': pm_tensor[6:7, :, :],
                'normal': pm_tensor[4:6, :, :], 'fake': torch.zeros(1, pm_tensor.shape[1], pm_tensor.shape[2], dtype=torch.float32),
                'ans_padded': ans_padded}

    # --- get curve features
    ans_dict['normOD'] = norm_list
    ans_dict['tangentOD'] = tangent_list
    arcnorm_list = []
    for _, a_tensor in enumerate(ans_list):
        this_nv = a_tensor.shape[0]
        arcnorm_feature = np.arange(this_nv) / float(this_nv - 1)  # 0 -> 1
        arcnorm_feature = arcnorm_feature.astype(np.float32)
        arcnorm_list.append(torch.from_numpy(arcnorm_feature[:, np.newaxis]))  # n, 1
    ans_dict['arcnormOD'] = arcnorm_list

    return ans_dict


def collect_con_feat(vg_dict, feat_str):
    ans_list = []
    feat_list = feat_str.split('_')
    if len(feat_list) == 1:
        return vg_dict[feat_list[0]]
    else:
        for feat_s_str in feat_list:
            ans_list.append(vg_dict[feat_s_str])
        return torch.cat(ans_list, 0)


def get_svg_shapes(vg_dict, thickness_list=None, hw=768, dpxy_list=None):
    shapes = []
    shape_groups = []
    new_id = 0

    for path_id, points_tensor in enumerate(vg_dict['ans_list']):  # creat svg

        ncp_tensor = torch.zeros(points_tensor.shape[0] - 1, dtype=torch.int32).cuda()

        if dpxy_list is not None:
            dp_v = dpxy_list[path_id].permute(1, 0)  # nv, 2  # absolute dp
            new_pos_tensor = points_tensor + dp_v
        else:
            new_pos_tensor = points_tensor

        if thickness_list is None:
            new_t_tensor = torch.ones(points_tensor.shape[0], dtype=torch.float32).to(points_tensor.device)
        else:
            new_t_tensor = thickness_list[new_id]

        path = pydiffvg.Path(num_control_points=ncp_tensor,
                             points=new_pos_tensor,
                             is_closed=False,
                             stroke_width=new_t_tensor,
                             id='path_%d' % new_id)
        shapes.append(path)

        path_group = pydiffvg.ShapeGroup(shape_ids=torch.tensor([new_id]).cuda(),
                                         fill_color=None,
                                         stroke_color=torch.tensor([0, 0, 0, 1]).cuda(),
                                         use_even_odd_rule=False)
        shape_groups.append(path_group)
        new_id += 1

    scene_args = pydiffvg.RenderFunction.serialize_scene(hw, hw, shapes, shape_groups)
    return scene_args


def get_pm_attributes_from_1D(pm_dict, feat_1d, min_thickness=0.5):  # no activation    1/2, 64, 64    dp then thickness

    th_list = []
    dp_list = []
    for path_id, indices_tensor in enumerate(pm_dict['indices_list']):
        this_nv = indices_tensor.shape[0]
        propagated_code = feat_1d[path_id, :, 0:this_nv]  # 3, this-nv

        th_list.append(F.leaky_relu(propagated_code[-1, :]) + min_thickness)  # definition of real activation function

        dp_list.append(propagated_code[0:2, :])  # absolute dp; no activation; 2, nv
    return dp_list, th_list


def remove_alpha(input_tensor, output_color=False):
    if not output_color:
        img_tensor = input_tensor[:, :, 3] * input_tensor[:, :, 0] + torch.ones(input_tensor.shape[0], input_tensor.shape[1], dtype=torch.float32).cuda() * (1 - input_tensor[:, :, 3])
    else:
        img_tensor = input_tensor[:, :, 3:4] * input_tensor[:, :, 0:3] + torch.ones(input_tensor.shape[0], input_tensor.shape[1], 3, dtype=torch.float32).cuda() * (1 - input_tensor[:, :, 3:4])
    return img_tensor


def resample_curves(ifn, ofn):

    with open(ifn, 'rb') as f:
        temp_list = pickle.load(f)  # a list of curves

    input_list = []
    for a_curve in temp_list:
        new_curve = []
        for a_vertex in a_curve:
            new_curve.append(a_vertex[0:2])
        input_list.append(new_curve)

    output_list = []
    for a_curve in input_list:
        temp_instance = PMPolyLine(a_curve)  # a curve instance
        temp_curve = temp_instance.resample_curve()
        output_list.append(temp_curve)


    ans = []
    for cid, curve_instance in enumerate(output_list):
        ans_stroke = []
        for id_vertex, a_vertex in enumerate(curve_instance):
            ans_stroke.append([float(a_vertex[0]), float(a_vertex[1])] + [0.0, 0.0, 0.0, 0.0, 0.0])
        ans.append(ans_stroke)
    with open(ofn, 'wb') as f:
        pickle.dump(ans, f, protocol=0)


class PMPolyLine(object):
    def __init__(self, a_stroke):

        self.vertices_list = []

        for a_id, a_vertice in enumerate(a_stroke):
            if len(self.vertices_list) > 0 and round(self.vertices_list[-1][0]) == round(a_vertice[0]) and round(self.vertices_list[-1][1]) == round(a_vertice[1]):  # duplicate
                continue
            self.vertices_list.append(a_vertice[:2])

        assert len(self.vertices_list) >= 2
        self.vertices_np = np.array(self.vertices_list)
        self.arc_length = calculate_arc_length(self.vertices_np)
        self.arc_t = calculate_arc_t(self.vertices_np)

    def resample_curve(self, threshold=3.0):  # arc length threshold
        ans_list = [[float(self.vertices_np[0, 0]), float(self.vertices_np[0, 1])]]
        current_length = 0.0
        current_length += threshold
        while True:
            if current_length > self.arc_length:
                break
            interpolated_x = np.interp(current_length, self.arc_t, self.vertices_np[:, 0])
            interpolated_y = np.interp(current_length, self.arc_t, self.vertices_np[:, 1])
            ans_list.append([float(interpolated_x), float(interpolated_y)])
            current_length += threshold
        return ans_list


def calculate_arc_t(vertices_np):
    nv = vertices_np.shape[0]
    assert nv >= 2
    ans = np.zeros((nv, ), dtype=np.float64)
    ans[1:] = np.cumsum(np.sqrt(np.sum((vertices_np[:-1] - vertices_np[1:]) ** 2, axis=1)))
    return ans


def calculate_arc_length(vertices_np):
    nv = vertices_np.shape[0]
    assert nv >= 2
    ans = np.sum(np.sqrt(np.sum((vertices_np[:-1] - vertices_np[1:]) ** 2, axis=1)))
    return ans
