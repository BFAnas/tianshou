#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

setup(
    name='tianshou',
    version='0.2.0',
    description='A Library for Deep Reinforcement Learning',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/thu-ml/tianshou',
    author='TSAIL',
    author_email='trinkle23897@gmail.com',
    license='MIT',
    python_requires='>=3.6',
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: MIT License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    keywords='reinforcement learning platform',
    packages=find_packages(exclude=['test', 'test.*',
                                    'examples', 'examples.*',
                                    'docs', 'docs.*']),
    install_requires=[
        'gym>=0.15.0',
        'tqdm',
        'numpy',
        'cloudpickle',
        'tensorboard',
        'torch>=1.4.0',
    ],
    extras_require={
        'atari': [
            'atari_py',
        ],
        'mujoco': [
            'mujoco_py',
        ]
    },
)
