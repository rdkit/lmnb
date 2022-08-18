#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

setup(
    author="Bartosz Baranowski",
    author_email="bartosz.baranowski@novartis.com",
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Programming Language :: Python :: 3.8",
    ],
    install_requires=[
        "pandas>=1.4.2",
        "numpy>=1.22.4",
        "scikit-learn>=1.1.1",
        "scipy>=1.8.1",
    ],
    extras_require={"test": ["pytest>=7.1.2", "rdkit>=2022.03.3"]},
    description="Naive Bayes implementation",
    entry_points={},
    include_package_data=True,
    name="laplacianNB",
    packages=find_packages(include=["bayes", "bayes.*"]),
    test_suite="tests",
    url="",  # noqa: E501
    version="0.7",
)
