"""
.. module:: args_parser
    :synopsis: unified argument parser

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Mar 2, 2018
"""

import argparse
import os


class ArgsParser:

    def __init__(self):

        self.args_dict = {

            'model': {
                'arg': '--model',
                'action': 'store',
                'dest': 'model',
                'default': 'oea_confusion_matrix_all',
                'help': 'Models are available in /conf. ' +
                        'Argument is the config filename suffix. ' +
                        '(e.g. --model oea_all # for config_oea_all config file).'
            },

            'dry_run': {
                'arg': '--dry_run',
                'action': 'store_true',
                'dest': 'dry_run',
                'default': False,
                'help': 'When enabled, the trained model won\'t be saved.'
            },

            'parallel': {
                'arg': '--parallel',
                'action': 'store_true',
                'default': False,
                'dest': 'parallel',
                'help': 'Enable parallel processing for faster results.'
            },

            'batch': {
                'arg': '--batch',
                'action': 'store_true',
                'default': False,
                'dest': 'batch',
                'help': 'Enable batch processing.'
            },

            'montage': {
                'arg': '--montage',
                'action': 'store_true',
                'default': False,
                'dest': 'montage',
                'help': 'Embed result on the image.'
            },

            'single_path': {
                'arg': '--single_path',
                'action': 'store',
                'dest': 'single_path',
                'default': os.path.join(
                    os.getcwd(),
                    'training_p2m',
                    'data',
                    'testset',
                    'happiness',
                    'img17.jpg'
                ),
                'help': 'Single image path if batch is disabled.'
            },

            'score': {
                'arg': '--score',
                'action': 'store_true',
                'dest': 'score',
                'default': False,
                'help': 'Add model accuracy score to output.'
            },

            'extract_to': {
                'arg': '--extract_to',
                'action': 'store',
                'dest': 'extract_to',
                'default': '',
                'help': 'Extract each tagged image.'
            }
        }

    def custom_parser(self, args_list):

        parser = argparse.ArgumentParser()

        for arg in args_list:

            d_ = self.args_dict[arg]

            parser.add_argument(
                d_['arg'],
                action=d_['action'],
                default=d_['default'],
                dest=d_['dest'],
                help=d_['help']
            )

        return parser
