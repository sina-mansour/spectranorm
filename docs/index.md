# SpectraNorm
A Python package for spectral normative modeling of high-dimensional data.

[![PyPI](https://img.shields.io/pypi/v/spectranorm?style=flat-square)](https://pypi.python.org/pypi/spectranorm/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/spectranorm?style=flat-square)](https://pypi.python.org/pypi/spectranorm/)
[![PyPI - License](https://img.shields.io/pypi/l/spectranorm?style=flat-square)](https://raw.githubusercontent.com/sina-mansour/spectranorm/refs/heads/main/LICENSE)
[![Coookiecutter - Wolt](https://img.shields.io/badge/cookiecutter-Wolt-00c2e8?style=flat-square&logo=cookiecutter&logoColor=D4AA00&link=https://github.com/woltapp/wolt-python-package-cookiecutter)](https://github.com/woltapp/wolt-python-package-cookiecutter)

## Installation
You can install the package using pip:

```sh
pip install spectranorm --upgrade
```

Installation typically takes less than 1 minute on a standard laptop with an internet connection.

## Requirements

SpectraNorm is implemented in Python and depends on several standard scientific Python libraries, including:

- numpy
- scipy
- pandas
- nibabel
- joblib
- pymc

All dependencies are automatically installed via pip.

While SpectraNorm should work in any Python 3.10+ environment, it has been specifically tested on a conda environment on a Linux machine running Python 3.12.

## Getting Started

Check out the [tutorials](tutorials/index.md) to get started with using SpectraNorm.

## API Reference

The [API reference](api/index.md) provides detailed documentation of several functions and classes available in the package.

## Dig Deeper

For more in-depth information about the underlying theory and example uses of spectral normative modeling, check out [this GitHub repository](https://github.com/sina-mansour/normative_brain_charts) which hosts several notebooks that implement spectral normative modeling of cortical thickness phenotypes on a large scale population-wide imaging biobank. You may also be interested in the [original paper](https://doi.org/10.1101/2025.01.16.25320639) describing the method and its application to brain imaging data.
