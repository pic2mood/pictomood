# Pictomood ![Size](https://github-size-badge.herokuapp.com/pic2mood/pictomood.svg)

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

- Python 3.6
- `requirements.txt`

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

- External APIs:
   - [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)
