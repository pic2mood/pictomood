# Pictomood ![Build Status](https://travis-ci.org/pic2mood/pictomood.svg) ![Size](https://github-size-badge.herokuapp.com/pic2mood/pictomood.svg) [![Maintainability](https://api.codeclimate.com/v1/badges/d2b65b4f93c3ab0234dc/maintainability)](https://codeclimate.com/github/pic2mood/pictomood/maintainability)

**Pictomood produces emotions out of images.**

## What?

**Pictomood is an implementation of the Object-to-Emotion Association (OEA) model, which adds object annotations as features to consider in predicting human emotion response towards an image.**

**Makes use of the following features:**

| Feature | Derived From
| - | -
| Object annotations | [Microsoft COCO: Common Objects in Context (Lin et. al., 2015)](http://arxiv.org/abs/1405.0312)
| Colorfulness score | [Measuring colourfulness in natural images (Hasler and  Susstrunk, 2003)](https://infoscience.epfl.ch/record/33994/files/HaslerS03.pdf)
| Dominant colors palette | [Color Thief](https://github.com/fengsp/color-thief-py)
| Mean GLCM contrast | [Textual Features for Image Classificiation (Haralick et. al., 1973)](http://haralick.org/journals/TexturalFeatures.pdf)

**Built on top of [scikit-learn](https://github.com/scikit-learn/scikit-learn) and [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection).**

## Dependencies

- [Python 3.6](https://www.python.org/)
- Python packages at `requirements.txt`

      # Available on both conda and pip
      scikit-image==0.13.0
      scikit-learn==0.19.1
      tensorflow==1.3.0
      pillow==5.0.0
      pandas==0.20.1
      numpy<=1.12.1
      opencv-python
      imutils

      # Available on pip only
      colorthief==0.2.1

- Repo-included APIs:
   - [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)

## Contributing

### Setup

1. Fork this repo.
2. Clone the fork.
3. Clone the [dataset](https://github.com/pic2mood/training_p2m.git) ![Size](https://github-size-badge.herokuapp.com/pic2mood/training_p2m.svg) to the clone's root path.

```bash
# BEFORE
pictomood # clone's root path, clone dataset here
L .git
L pictomood
L # other repo files
```

```bash
# TO CLONE,
$ git clone https://github.com/pic2mood/training_p2m.git {clone's root path}
```

```bash
# AFTER
pictomood # clone's root path, clone dataset here
L .git
L pictomood
L training_p2m # dataset clone path
L # other repo files
```

4. Setup Python environment.
5. Install dependencies.

```bash
$ pip install -r requirements.txt
```

### Run
#### Typical usage
```bash
python -m pictomood.pictomood --montage --score
```
#### Help
```bash
$ python -m pictomood.pictomood --help
usage: pictomood.py [-h] [--model MODEL] [--parallel] [--batch] [--montage]
                    [--single_path SINGLE_PATH] [--score]

optional arguments:
  -h, --help            show this help message and exit
  --model MODEL         Models are available in /conf. Argument is the config
                        filename suffix. (e.g. --model oea_all # for
                        config_oea_all config file).
  --parallel            Enable parallel processing for faster results.
  --batch               Enable batch processing.
  --montage             Embed result on the image.
  --single_path SINGLE_PATH
                        Single image path if batch is disabled.
  --score               Add model accuracy score to output.
```

### Train
#### Typical usage
```bash
python -m pictomood.trainer --model oea_all
```
#### Help
```bash
$ python -m pictomood.trainer oea --help
usage: trainer.py [-h] [--model MODEL] [--dry_run]

optional arguments:
  -h, --help     show this help message and exit
  --model MODEL  Models are available in /conf. Argument is the config filename
                 suffix. (e.g. --model oea_all # for config_oea_all config
                 file).
  --dry_run      When enabled, the trained model won't be saved.
```

## Authors

| [<img src="https://avatars1.githubusercontent.com/u/23053494?s=460&v=4" title="raymelon" width="80" height="80"><br/><sub>raymelon</sub>](https://github.com/raymelon)</br> | [<img src="https://avatars2.githubusercontent.com/u/27953463?s=460&v=4" title="gorejuice" width="80" height="80"><br/><sub>gorejuice</sub>](https://github.com/gorejuice)</br> |
| :---: | :---: |

