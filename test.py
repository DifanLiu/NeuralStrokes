"""
Copyright (C) 2021 Adobe. All rights reserved.
"""

import os
from options.test_options import TestOptions
from data import create_dataset_condition
from models import create_model
import torch
from util.utilNS import show_img, txt2list, mask_stroke_texture
import argparse


if __name__ == '__main__':
    '''
    SG: Stroke Geometry
    ST: Stroke Texture
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', '-d', type=str, required=True, help='path to testing data.')
    parser.add_argument('--ckpt_SG_path', '-g', type=str, required=True, help='path to Stroke Geometry checkpoints.')
    parser.add_argument('--ckpt_ST_path', '-t', type=str, required=True, help='path to Stroke Texture checkpoints.')
    parser.add_argument('--save_path', '-s', type=str, required=True, help='path to save synthesized image.')
    args = parser.parse_args()

    dataroot = args.dataroot
    SG_load_path = args.ckpt_SG_path
    ST_load_path = args.ckpt_ST_path
    save_path = args.save_path

    SG_config_fn = os.path.join(os.path.split(SG_load_path)[0], 'TEST_config_SG.txt')
    if not os.path.isfile(SG_config_fn):
        SG_config_fn = 'configs/TEST_config_SG.txt'
        assert os.path.isfile(SG_config_fn)
        print('Using default configuration file for Stroke Geometry prediction.')

    ST_config_fn = os.path.join(os.path.split(ST_load_path)[0], 'TEST_config_ST.txt')
    if not os.path.isfile(ST_config_fn):
        ST_config_fn = 'configs/TEST_config_ST.txt'
        assert os.path.isfile(ST_config_fn)
        print('Using default configuration file for Stroke Texture prediction.')

    #------- SG
    SG_config_suffix_list = ['--dataroot', dataroot]
    SG_opt = TestOptions(cmd_line=(txt2list(SG_config_fn) + SG_config_suffix_list)).parse()  # get test options

    dataset = create_dataset_condition(SG_opt)  # create a dataset given opt.dataset_mode and other options
    SG_model = create_model(SG_opt)  # create a model given opt.model and other options
    assert os.path.isfile(SG_load_path)

    # ------- ST
    ST_opt = TestOptions(cmd_line=txt2list(ST_config_fn)).parse()  # get test options

    ST_model = create_model(ST_opt)  # create a model given opt.model and other options
    assert os.path.isfile(ST_load_path)

    with torch.no_grad():
        SG_model.setup(SG_opt)  # regular setup: load and print networks; create schedulers
        SG_model.load_networks(SG_load_path)
        SG_model.eval()
        data = dataset[0]
        SG_model.set_input_test(data)
        SG_model.forward()  # run inference

        data['A'] = SG_model.fake_B.cpu()

        ST_model.setup(ST_opt)  # regular setup: load and print networks; create schedulers
        ST_model.load_networks(ST_load_path)
        ST_model.eval()
        ST_model.set_input_test(data)
        ST_model.forward()  # run inference

        output_img = mask_stroke_texture(data['A'], ST_model.fake_B.cpu(), ST_opt.dilation_iteration)
        show_img(output_img, save_path)
        print('Saved to %s' % save_path)
















