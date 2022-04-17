from setuptools import setup
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='modularitypruning',
    version='1.1.3',
    package_dir={'modularitypruning': 'utilities'},
    packages=['modularitypruning'],
    url='https://github.com/ragibson/ModularityPruning',
    license='',
    author='Ryan Gibson',
    author_email='ryan.alex.gibson@gmail.com',
    description='Pruning tool to identify small subsets of network partitions that are '
                'significant from the perspective of stochastic block model inference.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only',
        'License :: OSI Approved :: MIT License'
    ],
    python_requires='>=3.8, <4',
    install_requires=['champ', 'louvain', 'matplotlib', "numpy", 'psutil',
                      'python-igraph', "scipy", 'seaborn', 'sklearn']

)
