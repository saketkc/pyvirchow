#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""The setup script."""

import sys
from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize
try:
    import numpy
except ImportError:
    sys.stderr.write('Requires numpy for installation.')
    sys.exit(1)

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

with open('requirements.txt') as reqs:
    requirements = reqs.readlines()

setup_requirements = ['numpy']

test_requirements = [
    'pytest',
]

setup(
    author="Saket Choudhary",
    author_email='saketkc@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: BSD License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description=
    "Python Boilerplate contains all the boilerplate you need to create a Python package.",
    entry_points={
        'console_scripts': [
            'pyvirchow=pyvirchow.cli:cli',
        ],
    },
    install_requires=requirements,
    license="BSD license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='pyvirchow',
    name='pyvirchow',
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/saketkc/pyvirchow',
    version='0.1.0',
    zip_safe=False,
    ext_modules=cythonize([
        Extension(
            'pyvirchow.segmentation._max_clustering_cython',
            ['pyvirchow/segmentation/_max_clustering_cython.pyx'],
            include_dirs=[numpy.get_include()])
    ]),
)
