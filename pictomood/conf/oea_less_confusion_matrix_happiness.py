
from pictomood.config import *

model = {
    'name': 'oea_less',
    'validator': 'confusion_matrix',
    'emotions': 'happiness',
    'ext': '.pkl'
}

trainer = {
    'dataset': path_as([
        'data',
        model['name'],
        name_generator(model, suffix='dataset')
    ]),
    'model': path_as([
        'data',
        model['name'],
        name_generator(model, suffix='model')
    ]),

    'raw_images_dataset': os.path.join(
        os.getcwd(),
        'training_p2m',
        'data',
        dir_generator(model)
    ),
    'raw_images_testset': os.path.join(
        os.getcwd(),
        'training_p2m',
        'data',
        dir_generator(model)
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
