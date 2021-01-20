#!/usr/bin/env python3
import codecs
import os

from setuptools import setup, find_packages


with open(r"README.md", encoding="utf8") as f:
    long_description = f.read()


def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()


def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")


setup(
    name="ukis-csmask",
    version=get_version(os.path.join("ukis_csmask", "__init__.py")),
    url="https://github.com/dlr-eoc/ukis-csmask",
    author="German Aerospace Center (DLR)",
    author_email="ukis-helpdesk@dlr.de",
    license="Apache 2.0",
    description="masks clouds and cloud shadows in Sentinel-2, Landsat-8, Landsat-7 and Landsat-5 images",
    zip_safe=False,
    packages=find_packages(),
    install_requires=open("requirements.txt").read().splitlines(),
    extras_require={"gpu": ["onnxruntime-gpu",], "dev": ["pytest",],},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    python_requires=">=3.7",
    long_description=long_description,
    long_description_content_type="text/markdown",
    include_package_data=True,
)
