"""
.. module:: trainer
    :synopsis: trainer module

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Dec 23, 2017
"""
from pictomood import config
from pictomood.utils import *

import argparse
import importlib


def train_emotion(trainer_, dry_run=False, combinations=None):

    if combinations is None:
        build_dataset(
            trainer=trainer_,
            dry_run=dry_run
        )
    else:
        build_dataset(
            trainer=trainer_,
            dry_run=dry_run,
            emotion_combinations=combinations
        )

    inputs = trainer_['columns'][1:-2]
    train(
        trainer=trainer_,
        inputs=inputs
    )

    mlp = MLP()
    mlp.load_model(path=trainer_['model'])

    df = pd.read_pickle(trainer_['dataset'])
    # print('Dataset:\n', df)
    df = df[inputs].as_matrix()
    for data in df:
        print('Input:', data)
        print('Run:', mlp.run(input_=data))


if __name__ == '__main__':

    # import sys

    # if len(sys.argv) > 1:

    #     if sys.argv[1] == 'oea':
    #         trainer = config.trainer_oea

    #     elif sys.argv[1] == 'oea_less':
    #         trainer = config.trainer_oea_less

    #     else:
    #         raise ValueError('Invalid argument {0}'.format(sys.argv[1]))

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model',
        action='store',
        dest='model',
        default='oea_all',
        help='Two pictomood models available: oea and oea_less.'
    )
    parser.add_argument(
        '--dry_run',
        action='store_true',
        default=False,
        dest='dry_run',
        help='When enabled, the trained model won\'t be saved.'
    )

    args = parser.parse_args()

    # if args.model == 'oea':
    #     trainer = config.trainer_oea

    # elif args.model == 'oea_less':
    #     trainer = config.trainer_oea_less

    trainer = importlib.import_module('pictomood.conf.config_' + args.model).trainer

    train_emotion(
        trainer_=trainer,
        dry_run=args.dry_run
    )
