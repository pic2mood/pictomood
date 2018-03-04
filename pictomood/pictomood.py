"""
.. module:: pictomood
    :synopsis: main module for pictomood package

.. moduleauthor:: Raymel Francisco <franciscoraymel@gmail.com>
.. created:: Nov 24, 2017
"""
from pictomood.imports import *
from pictomood import config
from pictomood.lib.mlp import MLP
from pictomood.utils import *
from pictomood import conf
from pictomood import args_parser

import multiprocessing as mp
import argparse
import importlib
from PIL import Image


class Pictomood:

    def __init__(
            self,
            trainer,
            parallel=False,
            batch=False,
            montage=False,
            single_path='',
            score=False,
            extract_to=''
    ):
        self.trainer = trainer
        self.mlp = MLP()
        self.mlp.load_model(path=self.trainer['model'])

        self.parallel = parallel

        if self.parallel:
            self.pool = mp.Pool()

        self.batch = batch
        self.montage = montage

        if not self.batch:
            self.single_path = single_path

        self.score = score
        self.extract_to = extract_to

    def run(self, img, img_path):

        input_ = []
        for key, func in self.trainer['features'].items():

            if key == 'top_colors':
                if self.parallel:
                    feature = self.pool.apply_async(
                        func,
                        [img, img_path]
                    ).get()

                else:
                    feature = func(img, img_path)

            elif key == 'top_object':
                # annotator module uses unpickable objects
                #   which can't be pooled
                feature = func(img)

            else:
                if self.parallel:
                    feature = self.pool.apply_async(
                        func,
                        [img]
                    ).get()

                else:
                    feature = func(img)

            # if multiple features in one category
            if isinstance(feature, collections.Sequence):
                for item in feature:
                    input_.append(item)
            else:
                input_.append(feature)

        return input_, self.mlp.run(input_=input_).tolist()

    def batch_process(self):

        if self.montage:
            to_montage = []

        emotion_combinations = ['happiness', 'sadness', 'fear']
        emotions = {}
        for em in emotion_combinations:
            emotions[em] = config.emotions_map[em]

        input_ = []
        output = []
        for emotion_str, emotion_val in emotions.items():

            dir_images = os.path.join(
                self.trainer['raw_images_testset'],
                # importlib.import_module('pictomood.conf.oea_confusion_matrix_all').trainer['raw_images_testset'],
                emotion_str
            )

            csv_data = []

            for i, (img, img_path) in enumerate(
                image_batch_loader(dir_=dir_images, limit=None)
            ):
                print(img_path)

                features, result = self.run(img, img_path)

                print('Result:', config.emotions_list[result[0]])
                print('Expected:', emotion_str)

                # csv_data.append([img_path, config.emotions_list[result[0]]])

                if self.score:
                    input_.append(features)
                    output.append(config.emotions_map[emotion_str])

                if self.montage or self.extract_to:
                    # embed the predicted result
                    img = copyMakeBorder(
                        img,
                        top=0,
                        bottom=200,
                        left=0,
                        right=0,
                        borderType=BORDER_CONSTANT,
                        value=(0, 0, 0)
                    )
                    put_text(
                        img=img,
                        text=config.emotions_list[result[0]],
                        offset=(40, 40),
                        color=(0, 255, 0)
                    )
                    if config.emotions_list[result[0]] == emotion_str:
                        # green if predicted correctly
                        color_ = (0, 255, 0)
                    else:
                        # red if incorrect
                        color_ = (255, 0, 0)

                    # embed the expected result
                    put_text(
                        img=img,
                        text=emotion_str,
                        offset=(40, 75),
                        color=color_
                    )
                    put_text(
                        img=img,
                        text='TC: ' + str(features[0]),
                        offset=(20, 85 + 200),
                        color=(0, 255, 0)
                    )
                    put_text(
                        img=img,
                        text='CS: ' + str(features[3]),
                        offset=(20, 115 + 200),
                        color=(0, 255, 0)
                    )
                    put_text(
                        img=img,
                        text='TX: ' + str(features[4]),
                        offset=(20, 145 + 200),
                        color=(0, 255, 0)
                    )

                    # to_montage.append(img)

                    if self.extract_to:

                        # get emotion folder and filename
                        extract_file = os.path.join(
                            *img_path.split('/')[-2:]
                        )

                        # append extract path to it
                        extract_file = os.path.join(
                            self.extract_to,
                            extract_file
                        )

                        print('Extract to:', extract_file)

                        extract_dir = '/' + os.path.join(
                            *extract_file.split('/')[:-1]
                        )

                        print(extract_dir)

                        if not os.path.exists(extract_dir):
                            os.makedirs(extract_dir)
                        
                        # extract the image to the generated file
                        Image.fromarray(img).save(extract_file)

                    elif self.montage:
                        to_montage.append(img)

        if self.montage:
            montage_ = montage(to_montage)
            show(montage_)

        if self.parallel:
            # release multiprocessing pool
            self.pool.close()

        if self.score:
            print('Score:', self.mlp.model.score(input_, output))

        # csv_df = pd.DataFrame(csv_data, columns=['Image Dir', 'System EE'])
        # csv_df.to_csv('test-retest.csv')

        return output

    def single_process(self):

        print(self.single_path)

        emotion_str = self.single_path.split('/')[-2]

        img = image_single_loader(self.single_path)
        features, result = self.run(img, self.single_path)

        print('Result:', config.emotions_list[result[0]])
        print('Expected:', emotion_str)

        if self.montage:
            # embed the predicted result
            put_text(
                img=img,
                text=config.emotions_list[result[0]],
                offset=(40, 40),
                color=(0, 255, 0)
            )
            if config.emotions_list[result[0]] == emotion_str:
                # green if predicted correctly
                color_ = (0, 255, 0)
            else:
                # red if incorrect
                color_ = (255, 0, 0)

            # embed the expected result
            put_text(
                img=img,
                text=emotion_str,
                offset=(40, 70),
                color=color_
            )

            show(img)

        if self.score:
            print('Score:', self.mlp.model.score([features], [result]))

        return config.emotions_list[result[0]]


def main(args_=None):

    if not args_:

        # parser = argparse.ArgumentParser()
        # parser.add_argument(
        #     '--model',
        #     action='store',
        #     dest='model',
        #     default='oea_all',
        #     help='Models are available in /conf. ' +
        #     'Argument is the config filename suffix. (e.g. --model oea_all # for config_oea_all config file).'
        # )
        # parser.add_argument(
        #     '--parallel',
        #     action='store_true',
        #     default=False,
        #     dest='parallel',
        #     help='Enable parallel processing for faster results.'
        # )
        # parser.add_argument(
        #     '--batch',
        #     action='store_true',
        #     default=False,
        #     dest='batch',
        #     help='Enable batch processing.'
        # )
        # parser.add_argument(
        #     '--montage',
        #     action='store_true',
        #     default=False,
        #     dest='montage',
        #     help='Embed result on the image.'
        # )
        # parser.add_argument(
        #     '--single_path',
        #     dest='single_path',
        #     default=os.path.join(
        #         os.getcwd(),
        #         'training_p2m',
        #         'data',
        #         'testset',
        #         'happiness',
        #         'img17.jpg'
        #     ),
        #     help='Single image path if batch is disabled.'
        # )
        # parser.add_argument(
        #     '--score',
        #     action='store_true',
        #     dest='score',
        #     default=False,
        #     help='Add model accuracy score to output.'
        # )

        # args_ = parser.parse_args()

        parser = args_parser.ArgsParser()

        args_ = parser.custom_parser(
            args_list=[
                'single_path',
                'score',
                'model',
                'parallel',
                'batch',
                'montage',
                'extract_to'
            ]
        ).parse_args()

        args = {
            'single_path': '',
            'score': True,
            'model': 'oea_all',
            'parallel': False,
            'batch': False,
            'montage': False,
            'extract_to': ''
        }

        args['model'] = args_.model
        args['parallel'] = args_.parallel
        args['batch'] = args_.batch
        args['montage'] = args_.montage
        args['single_path'] = args_.single_path
        args['score'] = args_.score,
        args['extract_to'] = args_.extract_to

    else:
        args = args_

    # if args['model'] == 'oea':
    #     trainer = config.trainer_oea

    # elif args['model'] == 'oea_less':
    #     trainer = config.trainer_oea_less

    trainer = importlib.import_module('pictomood.conf.' + args['model']).trainer

    enna = Pictomood(
        trainer=trainer,
        parallel=args['parallel'],
        batch=args['batch'],
        montage=args['montage'],
        single_path=args['single_path'],
        score=args['score'],
        extract_to=args['extract_to']
    )

    if args['batch']:
        result = enna.batch_process()
    else:
        result = enna.single_process()

    return result


if __name__ == '__main__':
    result = main()
    print('RESULT:', result)
