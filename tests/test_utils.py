import inspect
import numpy as np
import os
import pytest
import sys

# enable relative import of package
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from ukis_csmask.utils import reclassify, tile_array, untile_array


def test_reclassify():
    pass


def test_tile_array():
    pass


def test_untile_array():
    pass
