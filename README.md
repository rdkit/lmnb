# LaplacianNB

<div>

| | |
| --- | --- |
| Package | [![PyPI - Version](https://img.shields.io/pypi/v/laplaciannb.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/laplaciannbs/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/laplaciannb.svg?color=blue&label=Downloads&logo=pypi&logoColor=gold)](https://pypi.org/project/laplaciannb/) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/laplaciannb.svg?logo=python&label=Python&logoColor=gold)](https://pypi.org/project/laplaciannb/) |
| Meta | [![code style - black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![types - Mypy](https://img.shields.io/badge/types-Mypy-blue.svg)](https://github.com/python/mypy) [![imports - isort](https://img.shields.io/badge/imports-isort-ef8336.svg)](https://github.com/pycqa/isort) [![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause) |
| | |

</div>

**LaplacianNB** is a Python module developed at **Novartis AG** for a Naive Bayes classifier for Laplacian modified models based on the scikit-learn Naive Bayes implementation.

This classifier is suitable for binary/boolean data as it uses only indices of the positive bits for prediction. The algorithm was first implemented in Pipeline Pilot and KNIME.

## Features

- Naive Bayes classifier for Laplacian modified models
- Suitable for binary/boolean data
- Efficient prediction using indices of positive bits

## Installation

You can install the package using pip:

```sh
pip install laplaciannb
```


## Literature

```
Nidhi; Glick, M.; Davies, J. W.; Jenkins, J. L. Prediction of biological targets
for compounds using multiple-category Bayesian models trained on chemogenomics
databases. J. Chem. Inf. Model. 2006, 46, 1124– 1133,
https://doi.org/10.1021/ci060003g

Lam PY, Kutchukian P, Anand R, et al. Cyp1 inhibition prevents doxorubicin-induced cardiomyopathy
in a zebrafish heart-failure model. Chem Bio Chem. 2020:cbic.201900741.
https://doi.org/10.1002/cbic.201900741
```

## Authors

Author and maintainer: **Bartosz Baranowski** (bartosz.baranowski@novartis.com)

## Maintainers

- **Bartosz Baranowski** (bartosz.baranowski@novartis.com)
- **Edgar Harutyunyan** (edgar.harutyunyan_ext@novartis.com)

## Changelog

- `v0.6.0` - Move to pdm build
- `v0.5.0` - Initial release

## License

This project is licensed under the BSD 3-Clause License - see the [LICENSE](LICENSE) file for details.
