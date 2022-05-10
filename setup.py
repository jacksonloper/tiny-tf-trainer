from setuptools import setup

setup(
    name='tinytftrainer',
    author='Jackson Loper',
    version='0.0.1',
    description='tiny opinionated package for training tf networks in jupyter notebooks',
    packages=[
        'tinytftrainer',
    ],
    install_requires=[
        'tensorflow',
    ]
)
