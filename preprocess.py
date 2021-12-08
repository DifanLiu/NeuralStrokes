"""
Copyright (C) 2021 Adobe. All rights reserved.
"""

from util.utilNS import resample_curves, ss2vg, get_svg_shapes, remove_alpha, show_img, tensor2img, create_folder
from NSutil import ifn2mask_tensor, conv2d
import torch
import pydiffvg
import numpy as np
import os
import argparse
import glob


parser = argparse.ArgumentParser()
parser.add_argument('--style_dir', '-s', type=str, required=True, help='path to style directory')
args = parser.parse_args()
style_dir = args.style_dir

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

pydiffvg.set_print_timing(False)
pydiffvg.set_use_gpu(True)

render = pydiffvg.RenderFunction.apply

data_list = glob.glob(style_dir + '/*')
for data_dir in data_list:
    data_name = os.path.split(data_dir)[1]

    #------ Evenly sample (ES) the raw planar map (pm)
    ifn = os.path.join(data_dir, 'pm.pkl')
    ofn = os.path.join(data_dir, 'pm_ES.pkl')
    resample_curves(ifn, ofn)

    # ------ render raster planar map
    ifn = os.path.join(data_dir, 'pm_ES.pkl')
    ofn = os.path.join(data_dir, 'plain.png')
    pm_dict = ss2vg(ifn)
    features_tensor = pm_dict['features_tensor']
    n_strokes = features_tensor.shape[0]
    n_vertices = features_tensor.shape[2]

    scene_args = get_svg_shapes(pm_dict)
    style_img_tensor = remove_alpha(render(768, 768, 2, 2, 1, None, *scene_args))  # h, w
    show_img(tensor2img(style_img_tensor), ofn)

    # ------ compute the POOL of effective crop positions for training
    if 'train' not in data_name:
        continue
    ifn = os.path.join(data_dir, 'plain.png')
    output_dir = create_folder(os.path.join(data_dir, 'pool'))

    crop_size_list = [64, 128, 192, 256]  # four different scales
    center_size_list = [64 - 34, 128 - 68, 192 - 102, 256 - 136]

    for crop_id, crop_size in enumerate(crop_size_list):
        center_size = center_size_list[crop_id]
        start_idx = int((crop_size - center_size) / 2)

        weight_tensor = torch.zeros(1, 1, crop_size, crop_size, dtype=torch.float32).cuda()
        weight_tensor[0, 0, start_idx:(start_idx + center_size), start_idx:(start_idx + center_size)] = 1.0

        mask_tensor = ifn2mask_tensor(ifn, 1.0, 1.0)  # 0 as background
        conv_tensor = conv2d(mask_tensor, weight_tensor)  # smaller size ;  full_hw - crop_hw + 1
        conv_np = conv_tensor.squeeze().cpu().numpy()
        xarray, yarray = np.nonzero(conv_np)
        final_array = np.concatenate((xarray[:, np.newaxis], yarray[:, np.newaxis]), axis=1)

        final_array = final_array.astype(np.int16)
        np.random.shuffle(final_array)
        ofn = os.path.join(output_dir, str(crop_id) + '.npy')
        np.save(ofn, final_array)
