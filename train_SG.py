"""
Copyright (C) 2021 Adobe. All rights reserved.
"""

import time
import torch
from options.train_options import TrainOptions
from options.test_options import TestOptions
from data import create_dataset_condition
from models import create_model
from util.visualizer import Visualizer
import numpy as np
from random import shuffle
import os
from util.utilNS import txt2list, show_img, tensor2img
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', '-d', type=str, required=True, help='path to training data')
    parser.add_argument('--name', '-n', type=str, required=True, help='name of the experiment')
    parser.add_argument('--checkpoints_dir', '-c', type=str, default='./checkpoints')

    args = parser.parse_args()

    dataroot = args.dataroot
    name = args.name
    checkpoints_dir = args.checkpoints_dir

    config_fn = os.path.join(dataroot, 'TRAIN_config_SG.txt')
    if not os.path.isfile(config_fn):
        config_fn = 'configs/TRAIN_config_SG.txt'
        assert os.path.isfile(config_fn)
        print('Using default configuration file for Stroke Geometry training.')

    test_config_fn = os.path.join(os.path.split(config_fn)[0], 'TEST_config_SG.txt')
    assert os.path.isfile(test_config_fn)

    config_suffix_list = ['--dataroot', dataroot, '--name', name, '--checkpoints_dir', checkpoints_dir]
    opt = TrainOptions(cmd_line=(txt2list(config_fn) + config_suffix_list)).parse()   # get training options

    if not os.path.exists(opt.checkpoints_dir):
        os.makedirs(opt.checkpoints_dir)

    dataset = create_dataset_condition(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size = len(dataset)    # get the number of images in the dataset.
    id_list = list(np.arange(dataset_size))

    model = create_model(opt)      # create a model given opt.model and other options
    print('The number of training images = %d' % dataset_size)

    visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    opt.visualizer = visualizer
    total_iters = 0                # the total number of training iterations

    optimize_time = 0.1

    times = []
    # range(1, 8 + 8 + 1)
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time = time.time()  # timer for entire epoch
        iter_data_time = time.time()    # timer for data loading per iteration
        epoch_iter = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch

        shuffle(id_list)
        for i, real_index in enumerate(id_list):  # inner loop within one epoch
            data = dataset[real_index]

            iter_start_time = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time

            batch_size = data["A"].size(0)  # 16
            total_iters += batch_size
            epoch_iter += batch_size
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_start_time = time.time()
            if epoch == opt.epoch_count and i == 0:
                model.data_dependent_initialize(data)
                model.setup(opt)               # regular setup: load and print networks; create schedulers
                model.parallelize()
            model.set_input(data)  # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights
            if len(opt.gpu_ids) > 0:
                torch.cuda.synchronize()
            optimize_time = (time.time() - optimize_start_time) / batch_size * 0.005 + 0.995 * optimize_time

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses = model.get_current_losses()
                visualizer.print_current_losses(epoch, epoch_iter, losses, optimize_time, t_data)
                if opt.display_id is None or opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                print(opt.name)  # it's useful to occasionally show the experiment name on console
                save_suffix = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()

        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
        model.update_learning_rate()                     # update learning rates at the end of every epoch.

    #---
    SG_config_suffix_list = ['--dataroot', opt.dataroot]
    SG_opt = TestOptions(cmd_line=(txt2list(test_config_fn) + SG_config_suffix_list)).parse()  # get test options
    dataset = create_dataset_condition(SG_opt)  # create a dataset given opt.dataset_mode and other options
    with torch.no_grad():
        model.eval()
        data = dataset[0]
        model.set_input_test(data)  # unpack data from data loader
        model.forward()  # run inference
    show_img(tensor2img(model.fake_B.squeeze()), os.path.join(opt.dataroot, 'artist_drawing_mask_diffvg.png'))
