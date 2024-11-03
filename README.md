# LaplacianNB

<div>

| | |
| --- | --- |
| CI/CD | [![tests](https://github.com/rdkit/lmnb/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/rdkit/lmnb/actions/workflows/tests.yml) [![Build - lmnb](https://github.com/bbaranow/lmnb/actions/workflows/build-lmnb.yml/badge.svg)](https://github.com/bbaranow/lmnb/actions/workflows/build-lmnb.yml) |
| Package | [![PyPI - Version](https://img.shields.io/pypi/v/laplaciannb.svg?logo=pypi&label=PyPI&logoColor=gold)](https://pypi.org/project/laplaciannbs/) [![PyPI - Downloads](https://img.shields.io/pypi/dm/laplaciannb.svg?color=blue&label=Downloads&logo=pypi&logoColor=gold)](https://pypi.org/project/laplaciannb/) [![PyPI - Python Version](https://img.shields.io/pypi/pyversions/laplaciannb.svg?logo=python&label=Python&logoColor=gold)](https://pypi.org/project/laplaciannb/) |
| Meta | [![Hatch project](https://img.shields.io/badge/%F0%9F%A5%9A-Hatch-4051b5.svg)](https://github.com/pypa/hatch) [![code style - black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) [![types - Mypy](https://img.shields.io/badge/types-Mypy-blue.svg)](https://github.com/python/mypy) [![imports - isort](https://img.shields.io/badge/imports-isort-ef8336.svg)](https://github.com/pycqa/isort) [![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)

</div>


**LaplacianNB** is a Python module developed at **Novartis AG** for Naive Bayes classifier for laplacian modified models based on scikit-learn Naive Bayes implementation.

This classifier is suitable for binary/boolean data as it uses for prediction only indices of the positive bits. The algorithm was first implemented in Pipeline Pilot and KNIME.

Literature:

```
Nidhi; Glick, M.; Davies, J. W.; Jenkins, J. L. Prediction of biological targets
for compounds using multiple-category Bayesian models trained on chemogenomics
databases. J. Chem. Inf. Model. 2006, 46, 1124– 1133,
https://doi.org/10.1021/ci060003g

Lam PY, Kutchukian P, Anand R, et al. Cyp1 inhibition prevents doxorubicin-induced cardiomyopathy
in a zebrafish heart-failure model. Chem Bio Chem. 2020:cbic.201900741.
https://doi.org/10.1002/cbic.201900741
```

=======


Authors
--------

Author and maintainer: **Bartosz Baranowski** (bartosz.baranowski@novartis.com)


Installation
------------

User installation:

```pip install laplaciannb```



Changelog
---------
`v0.6.0` - Move to pdm build
`v0.5.0` - Initial release
