#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  4 20:03:43 2020

@author: afranio
"""

import setuptools
import os

with open('README.md') as f:
    README = f.read()

requirements = os.path.join(
    os.path.dirname(os.path.realpath(__file__)),
    'requirements.txt'
)

if os.path.isfile(requirements):
    with open(requirements) as f:
        install_requires = f.read().splitlines()
else:
    install_requires = []

setuptools.setup(
    author="BibMon developers",
    author_email="cc-bibmon@petrobras.com.br",
    name='bibmon',
    description='Library with routines for data-driven process monitoring.',
    license='Apache 2.0',
    version='1.1.1',
    long_description=README,
    long_description_content_type='text/markdown',
    url='https://github.com/petrobras/bibmon',
    packages=setuptools.find_packages(include=['bibmon','bibmon.*']),
    include_package_data=True,
    package_data={
        'bibmon': ['real_process_data/*.csv','tennessee_eastman/*.dat'],
    },
    python_requires=">=3.12",
    install_requires=install_requires
)
