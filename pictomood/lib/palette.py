"""
.. module:: palette
    :synopsis: gets dominant colors in an image

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Dec 28, 2017
"""
from pictomood.imports import *

from colorthief import ColorThief
from skimage import io

from pictomood.utils import interpolate


class Palette:

    def dominant_colors(img, img_path, colors=2):

        temp_file = os.path.join(
            os.getcwd(),
            '.temps',
            '.temp_{0}'.format(img_path.split('/')[-1])
        )
        io.imsave(temp_file, img)

        #palette = ColorThief(temp_file).get_palette(color_count=colors)
        #os.remove(temp_file)

        palette = ColorThief(temp_file).get_palette(color_count=colors)
        os.remove(temp_file)

        colors = ()

        for color in palette:
            r, g, b = color
            hex_ = '#{:02x}{:02x}{:02x}'.format(r, g, b)
            int_ = int(hex_[1:], 16)

            colors = colors + (interpolate(int_, place=0.000000001),)

        return colors
