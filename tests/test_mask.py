import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import ndimage

sys.path.insert(0, str(Path().resolve()))

from ukis_csmask.mask import CSmask
from ukis_csmask.utils import reclassify, cohen_kappa_score


@pytest.mark.parametrize(
    "img, band_order, nodata_value",
    [
        (np.empty((20, 20, 6), np.float32), ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"], None),
        (np.empty((20, 20, 8), np.float32), ["Green", "Red", "Blue", "NIR", "SWIR1", "SWIR2", "NIR2", "ETC"], None),
        (np.empty((20, 20, 6), np.float32), ["Green", "Red", "Blue", "NIR", "SWIR1", "SWIR2"], -666),
    ],
)
def test_csmask_init(img, band_order, nodata_value):
    CSmask(img=img, band_order=band_order, nodata_value=nodata_value)


@pytest.mark.parametrize(
    "img, band_order, nodata_value",
    [
        (3, ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"], None),
        (np.empty((20, 20), np.float32), ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"], None),
        (np.empty((20, 20, 3), np.float32), ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"], None),
        (np.empty((20, 20, 6), np.uint8), ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"], None),
        (np.empty((20, 20, 6), np.float32), ["Blue", "Green", "Yellow", "NIR", "SWIR1", "SWIR2"], None),
    ],
)
def test_csmask_init_raises(img, band_order, nodata_value):
    with pytest.raises(TypeError):
        CSmask(img=img, band_order=band_order, nodata_value=nodata_value)


@pytest.mark.parametrize(
    "data",
    [
        np.load(r"tests/testfiles/sentinel2.npz"),
        np.load(r"tests/testfiles/landsat8.npz"),
        np.load(r"tests/testfiles/landsat7.npz"),
        np.load(r"tests/testfiles/landsat5.npz"),
    ],
)
def test_csmask_csm(data):
    csmask = CSmask(img=data["img"], band_order=["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"])
    y_pred = csmask.csm
    y_true = reclassify(data["msk"], {"reclass_value_from": [0, 1, 2, 3, 4], "reclass_value_to": [2, 0, 0, 0, 1]})
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    kappa = round(cohen_kappa_score(y_true, y_pred), 2)
    assert kappa >= 0.75


@pytest.mark.parametrize(
    "data",
    [
        np.load(r"tests/testfiles/sentinel2.npz"),
        np.load(r"tests/testfiles/landsat8.npz"),
        np.load(r"tests/testfiles/landsat7.npz"),
        np.load(r"tests/testfiles/landsat5.npz"),
    ],
)
def test_csmask_valid(data):
    csmask = CSmask(img=data["img"], band_order=["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"])
    y_pred = csmask.valid
    y_true = reclassify(data["msk"], {"reclass_value_from": [0, 1, 2, 3, 4], "reclass_value_to": [0, 1, 1, 1, 0]})
    y_true_inverted = ~y_true.astype(bool)
    y_true = (~ndimage.binary_dilation(y_true_inverted, iterations=4).astype(bool)).astype(np.uint8)
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    kappa = round(cohen_kappa_score(y_true, y_pred), 2)
    assert kappa >= 0.75
