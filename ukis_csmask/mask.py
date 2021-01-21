import numpy as np
import onnxruntime
from pathlib import Path
from scipy import ndimage

from .utils import reclassify, tile_array, untile_array


class CSmask:
    """Segments clouds and cloud shadows in multi-spectral satellite images."""

    def __init__(
        self, img, band_order=None, nodata_value=None,
    ):
        """
        :param img: Input satellite image of shape (rows, cols, bands). (ndarray).
            Requires images of Sentinel-2, Landsat-8, -7 or -5 in Top of Atmosphere reflectance [0, 1].
            Requires image bands to include at least "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2".
            Requires image bands to be in approximately 30 m resolution.
        :param band_order: Image band order. (dict).
            >>> band_order = {0: "Blue", 1: "Green", 2: "Red", 3: "NIR", 4: "SWIR1", 5: "SWIR2"}
        :param nodata_value: Additional nodata value that will be added to valid mask. (num).
        """
        # consistency checks on input image
        if band_order is None:
            band_order = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]

        if isinstance(img, np.ndarray) is False:
            raise TypeError("img must be of type np.ndarray")

        if img.ndim != 3:
            raise TypeError("img must be of shape (rows, cols, bands)")

        if img.shape[2] < 6:
            raise TypeError("img must contain at least 6 spectral bands")

        if img.dtype != np.float32:
            raise TypeError("img must be in top of atmosphere reflectance with dtype float32")

        # consistency checks on band_order
        target_band_order = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
        if band_order != target_band_order:
            if all(elem in band_order for elem in target_band_order):
                # rearrange image bands to match target_band_order
                idx = np.array(
                    [
                        np.where(band == np.array(band_order, dtype="S"))[0][0]
                        for band in np.array(target_band_order, dtype="S")
                    ]
                )
                img = np.stack(np.asarray([img[:, :, i] for i in range(img.shape[2])])[idx], axis=2)
            else:
                raise TypeError(
                    "img must contain at least ['Blue', 'Green', 'Red', 'NIR', 'SWIR1', 'SWIR2'] spectral bands"
                )
        else:
            # use image bands as are
            img = np.stack(np.asarray([img[:, :, i] for i in range(img.shape[2])]), axis=2)

        self.img = img
        self.band_order = band_order
        self.nodata_value = nodata_value
        self.csm = self._csm()
        self.valid = self._valid()

    def _csm(self):
        """Computes cloud and cloud shadow mask with following class ids: 0=background, 1=clouds, 2=cloud shadows.

        :returns: cloud and cloud shadow mask. (ndarray).
        """
        # tile array
        x = tile_array(self.img, xsize=256, ysize=256, overlap=0.2)

        # standardize feature space
        x -= [0.19312, 0.18659, 0.18899, 0.30362, 0.23085, 0.16216]
        x /= [0.16431, 0.16762, 0.18230, 0.17409, 0.16020, 0.14164]

        # start onnx inference session and load model
        sess = onnxruntime.InferenceSession(str(Path(__file__).parent) + "/model.onnx")

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

    def _valid(self):
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
        valid_i = ~valid.astype(np.bool)
        valid = (~ndimage.binary_dilation(valid_i, iterations=4).astype(np.bool)).astype(np.uint8)

        if self.nodata_value is not None:
            # add image nodata pixels to valid pixel mask
            valid[(self.img[:, :, 0] == self.nodata_value)] = 0

        return valid
