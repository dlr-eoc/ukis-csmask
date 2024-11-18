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
        (np.empty((256, 256, 6), np.float32), ["Blue", "Green", "Red", "NIR", "SWIR16", "SWIR22"], None),
        (np.empty((256, 256, 8), np.float32), ["Green", "Red", "Blue", "NIR", "SWIR16", "SWIR22", "NIR2", "ETC"], None),
        (np.empty((256, 256, 6), np.float32), ["Green", "Red", "Blue", "NIR", "SWIR16", "SWIR22"], -666),
        (np.empty((256, 256, 4), np.float32), ["Red", "Green", "Blue", "NIR"], 0),
        (np.empty((256, 256, 5), np.float32), ["Red", "Green", "Blue", "NIR", "SWIR22"], 0),
    ],
)
def test_csmask_init(img, band_order, nodata_value):
    CSmask(img=img, band_order=band_order, nodata_value=nodata_value)


@pytest.mark.parametrize(
    "img, band_order, nodata_value",
    [
        (3, ["Blue", "Green", "Red", "NIR", "SWIR16", "SWIR22"], None),
        (np.empty((256, 256), np.float32), ["Blue", "Green", "Red", "NIR", "SWIR16", "SWIR22"], None),
        (np.empty((256, 256, 3), np.float32), ["Blue", "Green", "Red", "NIR", "SWIR16", "SWIR22"], None),
        (np.empty((256, 256, 6), np.uint8), ["Blue", "Green", "Red", "NIR", "SWIR16", "SWIR22"], None),
        (np.empty((256, 256, 6), np.float32), ["Blue", "Green", "Yellow", "NIR", "SWIR16", "SWIR22"], None),
        (np.empty((256, 256, 6), np.float32), None, None),
    ],
)
def test_csmask_init_raises(img, band_order, nodata_value):
    with pytest.raises(TypeError):
        CSmask(img=img, band_order=band_order, nodata_value=nodata_value)


@pytest.mark.parametrize(
    "img, band_order, nodata_value",
    [
        (np.empty((128, 128, 6), np.float32), ["Blue", "Green", "Red", "NIR", "SWIR16", "SWIR22"], None),
        (np.empty((64, 64, 6), np.float32), ["Green", "Red", "Blue", "NIR", "SWIR16", "SWIR22"], -666),
    ],
)
def test_csmask_init_warns(img, band_order, nodata_value):
    with pytest.warns(UserWarning):
        CSmask(img=img, band_order=band_order, nodata_value=nodata_value)


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "data",
    [
        np.load(r"tests/testfiles/sentinel2.npz"),
        np.load(r"tests/testfiles/landsat8.npz"),
        np.load(r"tests/testfiles/landsat7.npz"),
        np.load(r"tests/testfiles/landsat5.npz"),
    ],
)
def test_csmask_csm_6band(data):
    csmask = CSmask(img=data["img"], band_order=["Blue", "Green", "Red", "NIR", "SWIR16", "SWIR22"])
    y_pred = csmask.csm
    y_true = reclassify(data["msk"], {"reclass_value_from": [0, 1, 2, 3, 4], "reclass_value_to": [2, 0, 0, 0, 1]})
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    kappa = round(cohen_kappa_score(y_true, y_pred), 2)
    assert kappa >= 0.75


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "data",
    [
        np.load(r"tests/testfiles/sentinel2.npz"),
        np.load(r"tests/testfiles/landsat8.npz"),
        np.load(r"tests/testfiles/landsat7.npz"),
        np.load(r"tests/testfiles/landsat5.npz"),
    ],
)
def test_csmask_valid_6band(data):
    csmask = CSmask(img=data["img"], band_order=["Blue", "Green", "Red", "NIR", "SWIR16", "SWIR22"])
    y_pred = csmask.valid
    y_true = reclassify(data["msk"], {"reclass_value_from": [0, 1, 2, 3, 4], "reclass_value_to": [0, 1, 1, 1, 0]})
    y_true_inverted = ~y_true.astype(bool)
    y_true = (~ndimage.binary_dilation(y_true_inverted, iterations=4).astype(bool)).astype(np.uint8)
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    kappa = round(cohen_kappa_score(y_true, y_pred), 2)
    assert kappa >= 0.75


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "data",
    [
        np.load(r"tests/testfiles/sentinel2.npz"),
        np.load(r"tests/testfiles/landsat8.npz"),
        np.load(r"tests/testfiles/landsat7.npz"),
        np.load(r"tests/testfiles/landsat5.npz"),
    ],
)
def test_csmask_csm_4band(data):
    csmask = CSmask(img=data["img"], band_order=["Blue", "Green", "Red", "NIR"])
    y_pred = csmask.csm
    y_true = reclassify(data["msk"], {"reclass_value_from": [0, 1, 2, 3, 4], "reclass_value_to": [2, 0, 0, 0, 1]})
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    kappa = round(cohen_kappa_score(y_true, y_pred), 2)
    assert kappa >= 0.50


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize(
    "data",
    [
        np.load(r"tests/testfiles/sentinel2.npz"),
        np.load(r"tests/testfiles/landsat8.npz"),
        np.load(r"tests/testfiles/landsat7.npz"),
        np.load(r"tests/testfiles/landsat5.npz"),
    ],
)
def test_csmask_valid_4band(data):
    csmask = CSmask(img=data["img"], band_order=["Blue", "Green", "Red", "NIR"])
    y_pred = csmask.valid
    y_true = reclassify(data["msk"], {"reclass_value_from": [0, 1, 2, 3, 4], "reclass_value_to": [0, 1, 1, 1, 0]})
    y_true_inverted = ~y_true.astype(bool)
    y_true = (~ndimage.binary_dilation(y_true_inverted, iterations=4).astype(bool)).astype(np.uint8)
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    kappa = round(cohen_kappa_score(y_true, y_pred), 2)
    assert kappa >= 0.50
