import json
import logging
import numpy as np
import os

from scipy import ndimage
from ukis_pysat.raster import Image
from ukis_pysat.members import Platform

from tensorflow.keras.models import load_model
from tensorflow.keras import backend as K
from .utils import (
    classification2binarymask,
    dice_coef,
    weighted_categorical_crossentropy,
    tile_array,
    untile_array,
    featurespace
)


class CSmask:
    """Segments clouds and cloud shadows in multi-spectral satellite images."""

    def __init__(
        self, img, band_order=["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"], nodata_value=None,
    ):
        """
        :param img: Input satellite image of shape (rows, cols, bands) (Ndarray).
            Requires images of Sentinel-2, Landsat-8, -7 or -5 in Top of Atmosphere reflectance [0, 1].
            Requires image bands to include at least "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2".
            Requires image bands to be in approximately 30 m resolution.
        :param band_order: Image band order (Dict).
            >>> band_order = {0: "Blue", 1: "Green", 2: "Red", 3: "NIR", 4: "SWIR1", 5: "SWIR2"}
        :param nodata_value: Additional nodata value that will be added to valid mask (Number).
        """
        # consistency checks on input image
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
        :returns: cloud and cloud shadow mask (ndarray)
        """
        # set model parameters for valid mask segmentation
        scaler_file = "./models/UNETMSB6A_VALID_scaler.pkl"
        weights_file = "./models/UNETMSB6A_VALID_classweights.npy"
        model_file = "./models/UNETMSB6A_VALID.h5"
        model = load_model(
            model_file,
            custom_objects={"dice_coef": dice_coef, "loss": weighted_categorical_crossentropy(np.load(weights_file))},
        )

        # tile array
        array_tiled = tile_array(self.img, xsize=256, ysize=256, overlap=0.2)

        if scaler_file:
            # standardize feature space with scaler
            X_tiled = featurespace(
                array_tiled, standardize=True, save_scaler=False, load_scaler=True, scaler_file=scaler_file
            )
        else:
            # use input array as is
            X_tiled = array_tiled

        # predict in small batches to keep memory under control
        # prob_tiled = model.predict(X_tiled, batch_size=10, verbose=1)
        # NOTE: this is a workaround to avoid memory leak in tensorflow model.predict as of version 2.2.0
        prob_tiled = np.empty((X_tiled.shape[0], X_tiled.shape[1], X_tiled.shape[2], 5), dtype=np.float32)
        bi = np.arange(start=0, stop=X_tiled.shape[0], step=10)
        bi = np.append(bi, X_tiled.shape[0])
        for index in np.arange(len(bi) - 1):
            batch_start = bi[index]
            batch_end = bi[index + 1]
            prob_tiled[batch_start:batch_end] = model.predict_on_batch(X_tiled[batch_start:batch_end])

        # untile the probabilities with smooth blending
        prob = untile_array(
            prob_tiled, (self.img.shape[0], self.img.shape[1], prob_tiled.shape[3]), overlap=0.2, smooth_blending=True
        )

        # compute argmax of probabilities to get class predictions
        pred = np.argmax(prob, axis=2).astype(np.uint8)

        #
        # clear keras session
        K.clear_session()

        return pred

    def _valid(self):
        """Converts the cloud shadow mask into a binary valid mask. This sets cloud and cloud shadow pixels to 0
        (invalid) and background to 1 (valid). Invalid pixels are buffered to reduce effect of cloud and shadow borders.
        Optionally image nodata values can be added to mask.
        :returns: binary valid mask (ndarray)
        """
        # reclassify cloud shadow mask to binary valid mask
        # Shadow (0), Cloud (4), Snow (2) -> not valid (0)
        # Water (1), Land (3) -> valid (1)
        class_dict = {"reclass_value_from": [0, 1, 2, 3, 4], "reclass_value_to": [0, 1, 0, 1, 0]}
        msk = classification2binarymask(self.csm, class_dict)

        # dilate the inverse of the binary valid pixel mask (invalid=0)
        # this effectively buffers the invalid pixels
        msk_i = ~msk.astype(np.bool)
        msk = (~ndimage.binary_dilation(msk_i, iterations=4).astype(np.bool)).astype(np.uint8)

        if self.nodata_value is not None:
            # add image nodata pixels to valid pixel mask
            msk[(self.img[:, :, 0] == msk[self.nodata_value])] = 0

        return msk
