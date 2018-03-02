
from pictomood.imports import *

import logging
import traceback
import pkg_resources

from pictomood.lib.palette import Palette
from pictomood.lib.mlp import MLP
from pictomood.lib.annotator import Annotator
from pictomood.lib.color import Color
from pictomood.lib.texture import Texture
# from pictomood.utils import *


logger_ = logging.getLogger(__name__)
logger_.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter(
    '\n[%(asctime)s] %(name)s: %(levelname)s: %(message)s'
)
handler.setFormatter(formatter)
logger_.addHandler(handler)


def log(message, verbose=True):
    def decorator(func):

        def wrapped(*args, **kwargs):
            logger_.info(message)
            try:
                func(*args, **kwargs)
            except Exception:
                if verbose:
                    logger_.error(traceback.format_exc())
            finally:
                logger_.info('DONE.')

        return wrapped
    return decorator


emotions_map = {
    'happiness': 1,
    'anger': 2,
    'surprise': 3,
    'disgust': 4,
    'sadness': 5,
    'fear': 6
}

emotions_list = [
    '',
    'happiness',
    'anger',
    'surprise',
    'disgust',
    'sadness',
    'fear'
]

verbose = True
as_package = True

try:
    pkg_resources.resource_filename(
        __name__,
        'data/' +
        'mscoco_label_map.pbtxt'
    )
except:
    as_package = False


def path_as(blobs):
    if as_package:
        path = pkg_resources.resource_filename(
            __name__,
            '/'.join(blobs)
        )
    else:
        path = os.path.join(
            os.getcwd(),
            *blobs
        )

    return path


annotator_params = {
    'model': 'ssd_mobilenet_v1_coco_11_06_2017',
    'ckpt': path_as([
        'data',
        'ssd_mobilenet_v1_coco_11_06_2017',
        'frozen_inference_graph.pb'
    ]),
    'labels': path_as([
        'data',
        'mscoco_label_map.pbtxt'
    ]),
    'classes': 90
}

annotator = Annotator(
    model=annotator_params['model'],
    ckpt=annotator_params['ckpt'],
    labels=annotator_params['labels'],
    classes=annotator_params['classes']
)

trainer_oea_less = {
    'dataset': path_as([
        'data',
        'oea_less_dataset.pkl'
    ]),
    'model': path_as([
        'data',
        'oea_less_model.pkl'
    ]),

    'raw_images_dataset': os.path.join(
        os.getcwd(),
        'training_p2m',
        'data',
        'dataset'
    ),
    'raw_images_testset': os.path.join(
        os.getcwd(),
        'training_p2m',
        'data',
        'testset_confusion_matrix'
        # 'testset'
    ),

    'features': {
        'top_colors': Palette.dominant_colors,
        'colorfulness': Color.scaled_colorfulness,
        'texture': Texture.texture
    },
    'columns': [
        'Image Path',
        'Top Color 1st',
        'Top Color 2nd',
        'Top Color 3rd',
        'Colorfulness',
        'Texture',
        'Emotion Tag',
        'Emotion Value'
    ]
}

