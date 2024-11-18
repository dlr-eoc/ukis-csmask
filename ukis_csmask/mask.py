import warnings

import json
import numpy as np
import scipy

from pathlib import Path

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
        product_level="l1c",
        nodata_value=None,
        invalid_buffer=4,
        intra_op_num_threads=0,
        inter_op_num_threads=0,
        providers=None,
        batch_size=1,
    ):
        """
        :param img: Input satellite image of shape (rows, cols, bands). (ndarray).
            Requires satellite images converted to reflectance values [0, 1].
            Requires image bands to include at least "Blue", "Green", "Red", "NIR" (uses 4 band model).
            For better performance requires image bands to include "Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2" (runs 6 band model).
            For better performance requires image bands to be in approximately 30 m resolution.
        :param band_order: Image band order. (list of string).
            >>> band_order = ["blue", "green", "red", "nir", "swir16", "swir22"]
        : param product_level: Image product level "l1c" or "l2a". (string).
            >>> product_level = "l2a"
        :param nodata_value: Additional nodata value that will be added to valid mask. (num).
        :param invalid_buffer: Number of pixels that should be buffered around invalid areas. (int).
        :param intra_op_num_threads: Sets the number of threads used to parallelize the execution within nodes. Default is 0 to let onnxruntime choose. (int).
        :param inter_op_num_threads: Sets the number of threads used to parallelize the execution of the graph (across nodes). Default is 0 to let onnxruntime choose. (int).
        :param providers: onnxruntime session providers. Default is None to let onnxruntime choose. (list).
            >>> providers = ["CUDAExecutionProvider", "OpenVINOExecutionProvider", "CPUExecutionProvider"]
        :param batch_size: Batch size. (int).
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

        # consistency checks on band_order and product_level
        if band_order is None:
            raise TypeError("band_order cannot be None")
        else:
            band_order = [b.lower() for b in band_order]
        product_level = product_level.lower()
        if product_level not in ["l1c", "l2a"]:
            raise TypeError("product_level must be 'l1c' or 'l2a'")

        # load model metadata
        if all(elem in band_order for elem in ["blue", "green", "red", "nir", "swir16", "swir22"]):
            model_file = str(Path(__file__).parent) + f"/model_6b_{product_level}.onnx"
        elif all(elem in band_order for elem in ["blue", "green", "red", "nir"]):
            model_file = str(Path(__file__).parent) + f"/model_4b_{product_level}.onnx"
        else:
            raise TypeError(
                f"band_order must contain at least 'blue', 'green', 'red', 'nir' "
                f"and for better performance also 'swir16' and 'swir22'"
            )
        with open(Path(model_file).with_suffix(".json")) as f:
            model_dict = json.load(f)

        # start onnx inference session and load model
        so = onnxruntime.SessionOptions()
        so.intra_op_num_threads = intra_op_num_threads
        so.inter_op_num_threads = inter_op_num_threads
        providers = onnxruntime.get_available_providers() if providers is None else providers
        self.session = onnxruntime.InferenceSession(model_file, sess_options=so, providers=providers)
        self.input_names = self.session.get_inputs()[0].name
        self.model_file = model_file
        self.band_order = band_order
        self.band_mean = model_dict["data"]["dataset_statistics"]["img"][0]
        self.band_std = model_dict["data"]["dataset_statistics"]["img"][1]
        self.nodata_value = nodata_value
        self.target_size = model_dict["data"]["target_size"]
        self.batch_size = batch_size
        self.model_version = model_dict["model"]["version"]

        # adjust band order and normalize image
        self.img = self.normalize(
            img=self.adjust_band_order(
                img=img,
                source_band_order=self.band_order,
                target_band_order=model_dict["data"]["dataset_statistics"]["band_names"],
            ).astype(np.float32),
            mean=self.band_mean,
            std=self.band_std,
        )

        # compute cloud shadow mask and valid mask
        self.csm = self._csm()
        self.valid = self._valid(invalid_buffer)

    @staticmethod
    def normalize(img, mean, std):
        img -= mean
        img /= std
        return img

    @staticmethod
    def adjust_band_order(img, source_band_order, target_band_order):
        if all(elem in source_band_order for elem in target_band_order) is False:
            raise TypeError(f"model_file requires the following image band_order {target_band_order}")
        idx = np.array(
            [
                np.where(band == np.array(source_band_order, dtype="S"))[0][0]
                for band in np.array(target_band_order, dtype="S")
            ]
        )
        return img[:, :, idx]

    @staticmethod
    def batch(iterable, batch_size=1):
        length = len(iterable)
        for ndx in range(0, length, batch_size):
            yield iterable[ndx : min(ndx + batch_size, length)]

    @staticmethod
    def softmax(logits):
        return scipy.special.softmax(logits, axis=1)

    def _csm(self):
        """Computes cloud and cloud shadow mask with following class ids: 0=background, 1=clouds, 2=cloud shadows.

        :returns: cloud and cloud shadow mask. (ndarray).
        """
        # tile array
        x = np.moveaxis(tile_array(self.img, xsize=self.target_size[0], ysize=self.target_size[1], overlap=0.2), -1, 1)

        # predict on array tiles
        y_prob = np.moveaxis(
            np.concatenate(
                [
                    self.softmax(self.session.run(None, {self.input_names: batch})[0])
                    for batch in self.batch(x, batch_size=self.batch_size)
                ]
            ),
            1,
            -1,
        )

        # untile probabilities with smooth blending
        y_prob = untile_array(
            y_prob, (self.img.shape[0], self.img.shape[1], y_prob.shape[3]), overlap=0.2, smooth_blending=True
        )

        return np.expand_dims(np.argmax(y_prob, axis=2).astype(np.uint8), axis=-1)

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
        valid = (~scipy.ndimage.binary_dilation(valid_i, iterations=invalid_buffer).astype(bool)).astype(np.uint8)

        if self.nodata_value is not None:
            # add image nodata pixels to valid pixel mask
            valid[(self.img[:, :, 0] == self.nodata_value)] = 0

        return valid
