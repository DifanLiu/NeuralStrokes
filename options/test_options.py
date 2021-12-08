"""
Copyright (C) 2021 Adobe. All rights reserved.
"""

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--phase', type=str, default='test', help='train, test')
        parser.add_argument('--dilation_iteration', default=-1, type=int, help='#iterations of stroke geometry dilation')

        self.isTrain = False

        return parser
