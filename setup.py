"""
Author: N. Abrate.

File: setup.py

Description: setup file for installing the pyNDUS package.
"""
from setuptools import find_packages
from distutils.core import setup

requirements = "requirements.txt"

setup(
   name='pyNDUS',
   version='0.0.1',
   author='Nicolo Abrate',
   author_email='nicolo.abrate@polito.it',
   url='https://github.com/nicoloabrate/pyNDUS',
   packages=find_packages(where="src"),
   package_dir={"": "src"},
   license='LICENSE.md',
   description='Perform python-based Nuclear Data Uncertainty and Sensitivity analyses (pyNDUS)',
   long_description=open('README.md').read(),
   long_description_content_type="text/markdown",
   test_suite="tests",
   setup_requires=['pytest-runner'],
   tests_require=['pytest'],
   include_package_data=True,
   classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS independent",
    ],
    install_requires=open(requirements).read().splitlines(),
    python_requires='>=3.8',
)
