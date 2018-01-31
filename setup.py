
from setuptools import setup, find_packages

setup(
    name='pictomood',
    version='0.2',
    description='Pictomood produces emotions out of images.',
    url='https://github.com/pic2mood/pictomood',
    author='Raymel Francisco',
    license='MIT',
    python_requires='~=3.6',
    packages=[
        'pictomood',
        'pictomood/api/object_detection',
        'pictomood/api/object_detection/utils',
        'pictomood/api/object_detection/protos',
        'pictomood/lib'
    ],
    entry_points={
        'console_scripts': [
            'pictomood = pictomood.pictomood:main'
        ]
    },
    install_requires=[
        'scikit-image==0.13.0',
        'scikit-learn==0.19.1',
        'tensorflow==1.3.0',
        'pillow==5.0.0',
        'pandas==0.20.1',
        'numpy<=1.12.1',
        'opencv-python',
        'imutils',
        'colorthief==0.2.1'
    ],
    include_package_data=True,
    package_data={'pictomood': [
        'data/*.pbtxt',
        'data/*.pkl',
        'data/ssd_mobilenet_v1_coco_11_06_2017/*.pb'
    ]}
)
