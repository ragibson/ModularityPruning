from setuptools import setup
import os

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='modularitypruning',
    version='1.2.3',
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
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only',
        'License :: OSI Approved :: MIT License'
    ],
    python_requires='>=3.7, <4',
    install_requires=["champ; python_version<'3.10'",
                      'louvain', 'matplotlib',
                      "numpy<1.22.0; python_version<'3.8'",
                      "numpy; python_version>='3.8'",
                      # TODO: louvain and leidenalg crash on python-igraph>=0.10
                      'psutil', 'python-igraph<0.10',
                      "scikit-learn; python_version>='3.8'",
                      "scikit-learn<1.1; python_version<'3.8'",
                      "scipy<1.8; python_version<'3.8'",
                      "scipy; python_version>='3.8'",
                      'seaborn']
)
