import warnings

import numpy as np

from pathlib import Path
from scipy import ndimage

from .utils import reclassify, tile_array, untile_array

try:
    import onnxruntime
except ImportError as e:
    msg = (
        "ukis-csmask dependencies are not installed.\n\n"
        "Please pip install as follows and specify your desired runtime provider [cpu], [openvino] or [gpu]:\n\n"
        "  python -m pip install ukis-csmask[cpu] --upgrade"
    )
    raise ImportError(str(e) + "\n\n" + msg)


class CSmask:
    """Segments clouds and cloud shadows in multi-spectral satellite images."""

    def __init__(
        self,
        img,
        band_order,
        nodata_value=None,
        invalid_buffer=4,
    ):
        """
        :param img: Input satellite image of shape (rows, cols, bands). (ndarray).
            Requires satellite images in Top of Atmosphere reflectance [0, 1].
            Requires image bands to include at least "Blue", "Green", "Red", "NIR" (uses 4 band model).
            For better performance requires image bands to include "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2" (runs 6 band model).
            For better performance requires image bands to be in approximately 30 m resolution.
        :param band_order: Image band order. (list of string).
            >>> band_order = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
        :param nodata_value: Additional nodata value that will be added to valid mask. (num).
        :param invalid_buffer: Number of pixels that should be buffered around invalid areas.
        """
        # consistency checks on input image
        if isinstance(img, np.ndarray) is False:
            raise TypeError("img must be of type np.ndarray")

        if img.ndim != 3:
            raise TypeError("img must be of shape (rows, cols, bands)")

        if img.shape[2] < 4:
            raise TypeError("img must contain at least 4 spectral bands")

        if img.dtype != np.float32:
            raise TypeError("img must be in top of atmosphere reflectance with dtype float32")

        if img.shape[0] < 256 or img.shape[1] < 256:
            warnings.warn(
                message=f"Your input image is smaller than the internal tiling size of 256x256 pixels. This may result "
                f"in suboptimal performance. Consider using a larger image size.",
                category=UserWarning,
            )

        # consistency checks on band_order
        if band_order is None:
            raise TypeError("band_order cannot be None")

        if all(elem in band_order for elem in ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]):
            target_band_order = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
            band_mean = [0.19312, 0.18659, 0.18899, 0.30362, 0.23085, 0.16216]
            band_std = [0.16431, 0.16762, 0.18230, 0.17409, 0.16020, 0.14164]
            model_file = str(Path(__file__).parent) + "/model_6b.onnx"
        elif all(elem in band_order for elem in ["Blue", "Green", "Red", "NIR"]):
            target_band_order = ["Blue", "Green", "Red", "NIR"]
            band_mean = [0.19312, 0.18659, 0.18899, 0.30362]
            band_std = [0.16431, 0.16762, 0.18230, 0.17409]
            model_file = str(Path(__file__).parent) + "/model_4b.onnx"
        else:
            raise TypeError(
                f"band_order must contain at least 'Blue', 'Green', 'Red', 'NIR' "
                f"and for better performance also 'SWIR1' and 'SWIR2'"
            )

        # rearrange image bands to match target_band_order
        idx = np.array(
            [np.where(band == np.array(band_order, dtype="S"))[0][0] for band in np.array(target_band_order, dtype="S")]
        )
        img = img[:, :, idx]

        self.img = img
        self.band_order = band_order
        self.band_mean = band_mean
        self.band_std = band_std
        self.nodata_value = nodata_value
        self.model_file = model_file
        self.csm = self._csm()
        self.valid = self._valid(invalid_buffer)

    def _csm(self):
        """Computes cloud and cloud shadow mask with following class ids: 0=background, 1=clouds, 2=cloud shadows.

        :returns: cloud and cloud shadow mask. (ndarray).
        """
        # tile array
        x = tile_array(self.img, xsize=256, ysize=256, overlap=0.2)

        # standardize feature space
        x -= self.band_mean
        x /= self.band_std

        # start onnx inference session and load model
        sess = onnxruntime.InferenceSession(self.model_file, providers=onnxruntime.get_available_providers())

        # predict on array tiles
        y_prob = [sess.run(None, {"input_1": tile[np.newaxis, :]}) for n, tile in enumerate(list(x))]
        y_prob = np.concatenate(y_prob)[:, 0, :, :, :]

        # untile probabilities with smooth blending
        y_prob = untile_array(
            y_prob, (self.img.shape[0], self.img.shape[1], y_prob.shape[3]), overlap=0.2, smooth_blending=True
        )

        # compute argmax of probabilities to get class predictions
        y = np.argmax(y_prob, axis=2).astype(np.uint8)

        # reclassify results
        class_dict = {"reclass_value_from": [0, 1, 2, 3, 4], "reclass_value_to": [2, 0, 0, 0, 1]}
        csm = reclassify(y, class_dict)

        return csm

    def _valid(self, invalid_buffer):
        """Converts the cloud and cloud shadow mask into a binary valid mask with following class ids:
        0=invalid (clouds, cloud shadows, nodata), 1=valid (rest). Invalid pixels are buffered to reduce effect of
        cloud and cloud shadow fuzzy boundaries. If CSmask was initialized with nodata_value it will be added to the
        invalid class.

        :returns: binary valid mask. (ndarray).
        """
        # reclassify cloud shadow mask to binary valid mask
        class_dict = {"reclass_value_from": [0, 1, 2], "reclass_value_to": [1, 0, 0]}
        valid = reclassify(self.csm, class_dict)

        # dilate the inverse of the binary valid pixel mask (invalid=0)
        # this effectively buffers the invalid pixels
        valid_i = ~valid.astype(bool)
        valid = (~ndimage.binary_dilation(valid_i, iterations=invalid_buffer).astype(bool)).astype(np.uint8)

        if self.nodata_value is not None:
            # add image nodata pixels to valid pixel mask
            valid[(self.img[:, :, 0] == self.nodata_value)] = 0

        return valid
