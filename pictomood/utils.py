"""
.. module:: utils
    :synopsis: utility functions module

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Dec 26, 2017
"""
import os
import pandas as pd
from PIL import Image
from glob import glob
from skimage import io, color
from scipy.misc import imresize
import numpy as np

import collections
from imutils import build_montages
from cv2 import (
    putText, FONT_HERSHEY_SIMPLEX, BORDER_CONSTANT,
    copyMakeBorder)

import matplotlib.pyplot as plt

from pictomood import config
from pictomood.lib.mlp import MLP


def image_single_loader(img_path):
    max_size = 300, 300
    img = Image.open(img_path)
    img.thumbnail(max_size, Image.ANTIALIAS)
    img = np.array(img)
    img = color.gray2rgb(img)

    return img


def image_path_loader(dir_, limit=None):

    # logger_.info('Test images dir: ' + dir_)
    print('Test images dir: ' + dir_)

    paths = []

    dir_glob = sorted(glob(os.path.join(dir_, '*.jpg')))

    for img_path in dir_glob[:limit]:
        paths.append(img_path)

    return paths


# @log('Loading images...')
def image_batch_loader(dir_, limit=None):

    # logger_.info('Test images dir: ' + dir_)
    print('Test images dir: ' + dir_)

    dir_glob = sorted(glob(os.path.join(dir_, '*.jpg')))

    for img_path in dir_glob[:limit]:

        img = image_single_loader(img_path)
        yield img, img_path


def interpolate(value, place=0.01):
    return float(format(value * place, '.3f'))


def show(img):
    Image.fromarray(img).show()


def montage(images):

    rows = 7
    max_cols = int(len(images) / rows)

    return build_montages(
        images,
        (180, 180),
        (rows, max_cols)
    )[0]


def put_text(img, text, offset, color: tuple):

    putText(
        img=img,
        text=text,
        org=offset,
        fontFace=FONT_HERSHEY_SIMPLEX,
        fontScale=0.8,
        color=color,
        thickness=3
    )


# @log('Initializing training...')
def train(trainer, inputs):

    df = pd.read_pickle(trainer['dataset'])

    mlp = MLP()
    mlp.load_model(path=None)

    inputs = df[inputs]
    outputs = df[['Emotion Value']].as_matrix().ravel()

    # logger_.info('Training fitness: ' + str(mlp.train(inputs, outputs)))
    mlp.train(inputs, outputs)
    mlp.save_model(path=trainer['model'])


def build_dataset(
    trainer, emotion_combinations=['happiness', 'sadness', 'fear']
):
    # emotion filtering
    emotions = {}
    for em in emotion_combinations:
        emotions[em] = config.emotions_map[em]

    # dataset building
    data = []
    for emotion_str, emotion_val in emotions.items():
        dir_images = os.path.join(trainer['raw_images_dataset'], emotion_str)

        for i, (img, img_path) in enumerate(
            image_batch_loader(dir_=dir_images, limit=None)
        ):
            datum = [img_path.split('/')[-1]]
            for key, func in trainer['features'].items():

                if key == 'top_colors':
                    feature = func(img, img_path)
                else:
                    feature = func(img)

                # if multiple features in one category
                if isinstance(feature, collections.Sequence):
                    for item in feature:
                        datum.append(item)
                else:
                    datum.append(feature)

            datum.extend([emotion_str, emotion_val])

            data.append(datum)

    # dataset saving
    df = pd.DataFrame(
        data,
        columns=trainer['columns']
    )
    config.logger_.debug('Dataset:\n' + str(df))
    df.to_pickle(trainer['dataset'])


def view_dataset(trainer):

    df = pd.read_pickle(trainer['dataset'])
    config.logger_.debug('Dataset:\n' + str(df))

    to_montage = []

    headers = list(df)
    mod_df = zip(*[df[h] for h in headers])

    for row in zip(mod_df):

        emotion, filename = row[0][7], row[0][0]

        img_path = os.path.join(
            trainer['raw_images_dataset'],
            emotion,
            filename
        )

        print(img_path)

        img = image_single_loader(img_path)

        img = copyMakeBorder(
            img,
            top=0,
            bottom=300,
            left=0,
            right=0,
            borderType=BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        put_text(
            img=img,
            text=emotion,
            offset=(20, 40 + 300),
            color=(0, 255, 0)
        )
        put_text(
            img=img,
            text='TC: ' + str(row[0][1]),
            offset=(20, 85 + 300),
            color=(0, 255, 0)
        )
        put_text(
            img=img,
            text='CS: ' + str(row[0][4]),
            offset=(20, 115 + 300),
            color=(0, 255, 0)
        )
        put_text(
            img=img,
            text='TX: ' + str(row[0][5]),
            offset=(20, 145 + 300),
            color=(0, 255, 0)
        )

        to_montage.append(img)

    montage_ = montage(to_montage)
    show(montage_)


def plot_dataset(trainer):

    def plot_emotion(df, emotion_tag, axis):

        df = df[df['Emotion Tag'] == emotion_tag]
        df = df[[
            'Image Path',
            # 'Top Color 1st',
            # 'Top Color 2nd',
            # 'Top Color 3rd',
            #'Colorfulness',
             'Texture'
        ]]

        graph = df.plot(
            ax=axis,
            kind='bar',
            title=emotion_tag,
            figsize=(15, 7),
            legend=True,
            fontsize=12,
            # style='o',
            grid=True
        )
        graph.set_xlabel('Image', fontsize=8)
        graph.set_ylabel('Features', fontsize=12)

    df = pd.read_pickle(trainer['dataset'])

    fig, axes = plt.subplots(nrows=2, ncols=2)
    fig.canvas.set_window_title('OEA Dataset')

    plot_emotion(df, 'happiness', axis=axes[0, 0])
    plot_emotion(df, 'sadness', axis=axes[0, 1])
    plot_emotion(df, 'fear', axis=axes[1, 0])

    plt.show()
