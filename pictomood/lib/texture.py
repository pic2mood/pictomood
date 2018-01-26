"""
.. module:: texture
    :synopsis: main module for texture feature extraction in images

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Nov 19, 2017
"""
import numpy as np

from skimage.feature import greycomatrix, greycoprops
from skimage import io, color

from pictomood.utils import interpolate


class Texture:

    def texture(img):
        img_gray = np.average(
            img,
            weights=[0.299, 0.587, 0.114],
            axis=2
        ).astype(np.uint8)
        # img_gray = color.rgb2gray(img).astype(np.uint8)

        greycomatrix_ = greycomatrix(

            img_gray,

            distances=[1, 2],
            angles=[0, 0.785398, 1.5708, 2.35619],
            #angles=[0],
            levels=256,
            normed=True,
            symmetric=True
        )
        greycomatrix_ = np.round(greycomatrix_, 3)

        texture = greycoprops(greycomatrix_, 'contrast')
        texture = np.mean(texture)

        texture = interpolate(value=texture, place=0.1)

        return texture
