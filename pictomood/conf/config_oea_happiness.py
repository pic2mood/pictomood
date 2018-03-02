
from pictomood.config import *

trainer = {
    'dataset': path_as([
        'data',
        'oea_happiness_dataset.pkl'
    ]),
    'model': path_as([
        'data',
        'oea_happiness_model.pkl'
    ]),

    'raw_images_dataset': os.path.join(
        os.getcwd(),
        'training_p2m',
        'data',
        'testset_confusion_matrix_happiness'
    ),
    'raw_images_testset': os.path.join(
        os.getcwd(),
        'training_p2m',
        'data',
        'testset_confusion_matrix_happiness',
    ),

    'features': {
        'top_colors': Palette.dominant_colors,
        'colorfulness': Color.scaled_colorfulness,
        'texture': Texture.texture,
        'top_object': annotator.annotate
    },
    'columns': [
        'Image Path',
        'Top Color 1st',
        'Top Color 2nd',
        'Top Color 3rd',
        'Colorfulness',
        'Texture',
        'Top Object',
        'Emotion Tag',
        'Emotion Value'
    ]
}
