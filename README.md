<p align="center">
  <img src="https://sina-mansour.github.io/spectranorm/assets/logo-with-text-horizontal.svg" alt="SpectraNorm" width="420">
</p>

<p align="center">
  <a href="https://pypi.python.org/pypi/spectranorm/"><img src="https://img.shields.io/pypi/v/spectranorm?style=flat-square" alt="PyPI"></a>
  <a href="https://pypi.python.org/pypi/spectranorm/"><img src="https://img.shields.io/pypi/pyversions/spectranorm?style=flat-square" alt="Python versions"></a>
  <a href="https://doi.org/10.5281/zenodo.22078039"><img src="https://zenodo.org/badge/DOI/10.5281/zenodo.22078039.svg" alt="DOI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-AGPLv3%20%2F%20Commercial-green?style=flat-square" alt="License"></a>
</p>

---

**Documentation**: [sina-mansour.github.io/spectranorm](https://sina-mansour.github.io/spectranorm)
&nbsp;&nbsp;|&nbsp;&nbsp;
**Source Code**: [github.com/sina-mansour/spectranorm](https://github.com/sina-mansour/spectranorm)
&nbsp;&nbsp;|&nbsp;&nbsp;
**PyPI**: [pypi.org/project/spectranorm](https://pypi.org/project/spectranorm/)

---

SpectraNorm is a Python package for **spectral normative modeling** of high-dimensional
data.

Conventional normative models fit one model per feature, which becomes intractable at
high spatial resolution. SpectraNorm instead fits the model in a spectral basis, using
eigenmodes to represent the signal compactly. A single fitted model then yields normative
ranges for arbitrary regions of interest, at any spatial scale, without refitting.

## Installation

```sh
pip install spectranorm --upgrade
```

Requires Python 3.10 or newer. Installation takes under a minute on a typical laptop.

## Getting Started

The [tutorials](https://sina-mansour.github.io/spectranorm/tutorials/) walk through
fitting a univariate normative model, constructing eigenmode bases, and fitting the full
spectral model. The [API reference](https://sina-mansour.github.io/spectranorm/api/)
documents the available classes and functions in detail.

## Applications

Studies that use SpectraNorm:

- **Mansour L., S., et al. (2026). Spectral Normative Modeling of Brain Structure. _medRxiv_.**
  DOI: [10.1101/2025.01.16.25320639](https://doi.org/10.1101/2025.01.16.25320639)
  &nbsp;·&nbsp;
  [Accompanying repository](https://github.com/sina-mansour/normative_brain_charts)

## Citing

If you use SpectraNorm, please cite the software:

> **Mansour L., S. (2026). SpectraNorm: a Python package for spectral normative modeling. _Zenodo_.**
> DOI: [10.5281/zenodo.22078039](https://doi.org/10.5281/zenodo.22078039)

along with the manuscript describing the method:

> **Mansour L., S., et al. (2026). Spectral Normative Modeling of Brain Structure. _medRxiv_.**
> DOI: [10.1101/2025.01.16.25320639](https://doi.org/10.1101/2025.01.16.25320639)

## License

SpectraNorm is **dual licensed**:

- **Non-commercial / Academic use**: [GNU AGPLv3](https://www.gnu.org/licenses/agpl-3.0.en.html)
- **Commercial use**: A separate commercial license is required

See the [LICENSE](LICENSE) file for full details.

## Contributing

Development setup, testing, and the release process are documented in
[RELEASING.md](RELEASING.md). In brief:

```sh
poetry install
poetry run pre-commit install
poetry run pytest
```

---

<sub>Generated using the <a href="https://github.com/woltapp/wolt-python-package-cookiecutter">wolt-python-package-cookiecutter</a> template.</sub>
