import inspect
import numpy as np
import os
import pytest
import sys

# enable relative import of package
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from ukis_csmask.mask import CSmask


def test_csmask_init():
    # correct inputs
    CSmask(img=np.empty((20, 20, 6), dtype=np.float32))
    CSmask(img=np.empty((20, 20, 6), dtype=np.float32), band_order=["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"])
    CSmask(
        img=np.empty((20, 20, 6), dtype=np.float32),
        band_order=["Green", "Red", "Blue", "NIR", "SWIR1", "SWIR2", "NIR2", "WHATEVER"],
    )
    CSmask(
        img=np.empty((20, 20, 6), dtype=np.float32),
        band_order=["Green", "Red", "Blue", "NIR", "SWIR1", "SWIR2"],
        nodata_value=-666,
    )

    # wrong inputs
    with pytest.raises(TypeError):
        CSmask(img=3)

    with pytest.raises(TypeError):
        CSmask(img=np.empty((20, 20), dtype=np.float32))

    with pytest.raises(TypeError):
        CSmask(img=np.empty((20, 20, 3), dtype=np.float32))

    with pytest.raises(TypeError):
        CSmask(img=np.empty((20, 20, 6), dtype=np.uint8))

    with pytest.raises(TypeError):
        CSmask(img=np.empty((20, 20, 6), dtype=np.uint8))

    with pytest.raises(TypeError):
        CSmask(
            img=np.empty((20, 20, 6), dtype=np.float32), band_order=["Blue", "Green", "Yellow", "NIR", "SWIR1", "SWIR2"]
        )


def test_csmask_csm():
    # run csm computation with all testfiles and compare to expected results (kappa > 0.9)
    pass


def test_csmask_valid():
    # run csm computation with all testfiles and compare to expected results (kappa > 0.9)
    pass
