import geojson
import numpy as np
import os
import pickle
import scipy.signal
import warnings

from tensorflow.keras import backend as K


import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib import gridspec


def classification2binarymask(classification, class_dict):
    """Reclassifies a classification array to a binary pixel mask according to the rules provided in class_dict.

    :param classification: Numpy array that hold categorical class values (Ndarray).
    :param class_dict: Dictionary that maps input class values to output class values (Dict). \n

    >>> class_dict = {
    >>>     "reclass_value_from":[0,1,2,3,4],
    >>>     "reclass_value_to":[0,1,0,0,0],
    >>> }

    :returns: Numpy array with binary [0,1] class values (Ndarray).
    """
    # reclassify to valid pixel binary mask
    binary_pixel_mask = np.zeros((classification.shape[0], classification.shape[1], 1), dtype=np.uint8)
    for i in range(len(class_dict["reclass_value_from"])):
        binary_pixel_mask[classification == class_dict["reclass_value_from"][i]] = class_dict["reclass_value_to"][i]

    return binary_pixel_mask.astype(np.uint8)


def dice_coef(y_true, y_pred):
    """Computes the Dice Coefficient.

    :param y_true: Numpy array that holds true categorical class values (Ndarray).
    :param y_pred: Numpy array that holds predicted categorical class values (Ndarray).
    :returns: Dice coefficient (Float).
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    smooth = 1.0
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def weighted_categorical_crossentropy(weights):
    """Computes the weighted categorical crossentropy loss.

    :param weights: Numpy array of shape (C,) where C is the number of classes (Ndarray).
    :returns: weighted categorical crossentropy (Float).

    Example: \n
    >>> # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
    >>> weights = np.array([0.5,2,10])
    >>> loss = weighted_categorical_crossentropy(weights)
    >>> model.compile(loss=loss,optimizer='adam')
    """
    weights = K.variable(weights)

    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss

    return loss


def featurespace(array, standardize=True, save_scaler=False, load_scaler=False, scaler_file=None):
    """Composes an image featurespace for a tiled array of shape (tiles, rows, cols, bands).

    :param array: Numpy array of shape (tiles, rows, cols, bands) (Ndarray).
    :param standardize: Apply data standardization (Boolean).
    :param save_scaler: Save scaler file (Boolean).
    :param load_scaler: Load scaler from file (Boolean).
    :param scaler_file: Scaler file (String).
    :returns: Reshaped and optionally standardized feature space (Ndarray).
    """
    X = []
    for i in range(array.shape[0]):
        band_list = []
        array_toa = array[i, :, :, :]
        for b in range(array_toa.shape[2]):
            band_list.append(array_toa[:, :, b])
        # put everything together
        X_ = np.dstack(band_list)
        X.append(X_)
    X = np.array(X)

    if standardize is True:
        # standardize feature space in batches of 10000
        # BUG: there is a bug in numpy related to large arrays with array_split() -> gives problems with reshape later
        # FIX: https://github.com/numpy/numpy/pull/11813/files
        # C:\Users\wiel_mc\AppData\Local\conda\conda\envs\pysatenv3\Lib\site-packages\numpy\lib\shape_base.py
        X_ = np.reshape(X, (X.shape[0] * X.shape[1] * X.shape[2], X.shape[3]))
        X_split = np.array_split(X_, int(X_.shape[0] / 100000.0), axis=0)

        if load_scaler is True:
            # load scaler from file
            with open(scaler_file[:-4] + ".pkl", "rb") as f:
                standard_scaler = pickle.load(f)
        else:
            # fit scaler on data
            standard_scaler = StandardScaler(copy=False)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for i in range(len(X_split)):
                    standard_scaler.partial_fit(X_split[i])

        # transform feature space
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for i in range(len(X_split)):
                X_split[i] = standard_scaler.transform(X_split[i], copy=False).astype(np.float16)
        X_scaled = np.vstack(X_split)
        X = np.reshape(X_scaled, X.shape)

        if save_scaler is True:
            # save scaler to file
            with open(scaler_file[:-4] + ".pkl", "wb") as file:
                pickle.dump(standard_scaler, file, protocol=2)

    return X


def rolling_window(array, window=(0,), asteps=None, wsteps=None, axes=None, toend=True):
    """Applies a rolling (moving) window to a ndarray.

    :param array: Array to which the rolling window is applied (array_like).
    :param window: Either a single integer to create a window of only the last axis or a
        tuple to create it for the last len(window) axes. 0 can be used as a to ignore a
        dimension in the window (int or tuple).
    :param asteps: Aligned at the last axis, new steps for the original array, ie. for
        creation of non-overlapping windows (tuple).
    :param wsteps: Steps for the added window dimensions. These can be 0 to repeat values
        along the axis (int or tuple (same size as window)).
    :param axes: If given, must have the same size as window. In this case window is
        interpreted as the size in the dimension given by axes. IE. a window
        of (2, 1) is equivalent to window=2 and axis=-2 (int or tuple)
    :param toend: If False, the new dimensions are right after the corresponding original
        dimension, instead of at the end of the array. Adding the new axes at the
        end makes it easier to get the neighborhood, however toend=False will give
        a more intuitive result if you view the whole array (bool).
    :returns: A view on `array` which is smaller to fit the windows and has windows added
        dimensions (0s not counting), ie. every point of `array` is an array of size
        window. (ndarray).
    """
    array = np.asarray(array)
    orig_shape = np.asarray(array.shape)
    window = np.atleast_1d(window).astype(int)

    if axes is not None:
        axes = np.atleast_1d(axes)
        w = np.zeros(array.ndim, dtype=int)
        for axis, size in zip(axes, window):
            w[axis] = size
        window = w

    # Check if window is legal:
    if window.ndim > 1:
        raise ValueError("`window` must be one-dimensional.")
    if np.any(window < 0):
        raise ValueError("All elements of `window` must be larger then 1.")
    if len(array.shape) < len(window):
        raise ValueError("`window` length must be less or equal `array` dimension.")

    _asteps = np.ones_like(orig_shape)
    if asteps is not None:
        asteps = np.atleast_1d(asteps)
        if asteps.ndim != 1:
            raise ValueError("`asteps` must be either a scalar or one dimensional.")
        if len(asteps) > array.ndim:
            raise ValueError("`asteps` cannot be longer then the `array` dimension.")
        # does not enforce alignment, so that steps can be same as window too.
        _asteps[-len(asteps) :] = asteps

        if np.any(asteps < 1):
            raise ValueError("All elements of `asteps` must be larger then 1.")
    asteps = _asteps

    _wsteps = np.ones_like(window)
    if wsteps is not None:
        wsteps = np.atleast_1d(wsteps)
        if wsteps.shape != window.shape:
            raise ValueError("`wsteps` must have the same shape as `window`.")
        if np.any(wsteps < 0):
            raise ValueError("All elements of `wsteps` must be larger then 0.")

        _wsteps[:] = wsteps
        _wsteps[window == 0] = 1
    wsteps = _wsteps

    # Check that the window would not be larger then the original:
    if np.any(orig_shape[-len(window) :] < window * wsteps):
        raise ValueError("`window` * `wsteps` larger then `array` in at least one dimension.")

    new_shape = orig_shape

    # For calculating the new shape 0s must act like 1s:
    _window = window.copy()
    _window[_window == 0] = 1

    new_shape[-len(window) :] += wsteps - _window * wsteps
    new_shape = (new_shape + asteps - 1) // asteps
    # make sure the new_shape is at least 1 in any "old" dimension (ie. steps
    # is (too) large, but we do not care.
    new_shape[new_shape < 1] = 1
    shape = new_shape

    strides = np.asarray(array.strides)
    strides *= asteps
    new_strides = array.strides[-len(window) :] * wsteps

    # The full new shape and strides:
    if toend:
        new_shape = np.concatenate((shape, window))
        new_strides = np.concatenate((strides, new_strides))
    else:
        _ = np.zeros_like(shape)
        _[-len(window) :] = window
        _window = _.copy()
        _[-len(window) :] = new_strides
        _new_strides = _

        new_shape = np.zeros(len(shape) * 2, dtype=int)
        new_strides = np.zeros(len(shape) * 2, dtype=int)

        new_shape[::2] = shape
        new_strides[::2] = strides
        new_shape[1::2] = _window
        new_strides[1::2] = _new_strides

    new_strides = new_strides[new_shape != 0]
    new_shape = new_shape[new_shape != 0]

    return np.lib.stride_tricks.as_strided(array, shape=new_shape, strides=new_strides)


def tile_array(array, xsize=512, ysize=512, overlap=0.1):
    """Splits an ndarray into equally sized tiles with overlap.

    :param array: Numpy array of shape (rows, cols, bands) (Ndarray).
    :param xsize: Xsize of tiles (Integer).
    :param ysize: Ysize of tiles (Integer).
    :param overlap: Overlap of tiles between 0.0 and 1.0 (Float).
    :returns: Numpy array of shape(tiles, rows, cols, bands) (Ndarray)
    """
    # get dtype, rows, cols, bands and dtype from first file
    dtype = array.dtype
    rows = array.shape[0]
    cols = array.shape[1]
    if array.ndim == 3:
        bands = array.shape[2]
    elif array.ndim == 2:
        bands = 1

    # get steps
    xsteps = int(xsize - (xsize * overlap))
    ysteps = int(ysize - (ysize * overlap))

    # pad array on all sides to fit all tiles.
    # replicate values here instead of filling with nan.
    # nan padding would cause issues for standardization and classification later on.
    ypad = ysize + 1
    xpad = xsize + 1
    array = np.pad(
        array,
        (
            (int(ysize * overlap), ypad + int(ysize * overlap)),
            (int(xsize * overlap), xpad + int(xsize * overlap)),
            (0, 0),
        ),
        mode="symmetric",
    )

    # tile the data into overlapping patches
    # this skips any tile at the end of row and col that exceeds the shape of the input array
    # therefore padding the input array is needed beforehand
    X_ = rolling_window(array, (xsize, ysize, bands), asteps=(xsteps, ysteps, bands))

    # access single tiles and write them to file and/or to ndarray of shape (tiles, rows, cols, bands)
    X = []
    for i in range(X_.shape[0]):
        for j in range(X_.shape[1]):
            X.append(X_[i, j, 0, :, :, :])

    return np.asarray(X, dtype=dtype)


def untile_array(array_tiled, target_shape, overlap=0.1, smooth_blending=False):
    """Untiles an ndarray back into the original image size.

    :param array_tiled: Numpy array of shape (tiles, rows, cols, bands) (Ndarray).
    :param target_shape: Target shape (rows, cols, bands) (List of Integer).
    :param overlap: Overlap of tiles between 0.0 and 1.0 (Float).
    :param smooth_blending: Apply smooth tile blending (Boolean).
    :returns: Numpy array of shape(rows, cols, bands) (Ndarray)
    """
    # get rows, cols, bands and dtype from first file
    dtype = array_tiled.dtype
    rows = target_shape[0]
    cols = target_shape[1]
    bands = target_shape[2]
    xsize = array_tiled.shape[1]
    ysize = array_tiled.shape[2]

    # use overlap to fit image size with fixed tile size
    xsteps = int(xsize - (xsize * overlap))
    ysteps = int(ysize - (ysize * overlap))

    # create target array
    # this needs to include any padding applied to the tiled array (same as in tile_array())
    array_target = np.zeros(target_shape)
    ypad = ysize + 1
    xpad = xsize + 1
    array_target = np.pad(
        array_target,
        (
            (int(ysize * overlap), ypad + int(ysize * overlap)),
            (int(xsize * overlap), xpad + int(xsize * overlap)),
            (0, 0),
        ),
        mode="symmetric",
    )

    # get xtiles and ytiles
    X_ = rolling_window(array_target, (xsize, ysize, bands), asteps=(xsteps, ysteps, bands))
    xtiles = int(X_.shape[0])
    ytiles = int(X_.shape[1])

    if smooth_blending is True:
        if overlap > 0.5:
            # for smooth blending to work overlap cannot be larger than 0.5.
            print("ERROR: Overlap needs to be <=0.5 when using smooth blending.")
            # exit()
        # define tapered cosine function (tukey) to be used for smooth blending
        window1d = scipy.signal.tukey(M=xsize, alpha=overlap * 2)
        window2d = np.expand_dims(np.expand_dims(window1d, axis=1), axis=2)
        window2d = window2d * window2d.transpose(1, 0, 2)

        # apply window spline 2d function to each tile
        array_tiled = np.array([tile * window2d for tile in array_tiled])

        # access single tiles and write them to target array
        t = 0
        xoffset = 0
        for x in range(xtiles):
            yoffset = 0
            for y in range(ytiles):
                array_target[
                    xoffset * xsteps : xoffset * xsteps + xsize, yoffset * ysteps : yoffset * ysteps + ysize, :
                ] = (
                    array_target[
                        xoffset * xsteps : xoffset * xsteps + xsize, yoffset * ysteps : yoffset * ysteps + ysize, :
                    ]
                    + array_tiled[t, :, :, :]
                )
                t += 1
                yoffset += 1
            xoffset += 1
    else:
        # access single tiles and write them to target array
        t = 0
        xoffset = 0
        for x in range(xtiles):
            yoffset = 0
            for y in range(ytiles):
                array_target[
                    xoffset * xsteps : xoffset * xsteps + xsize, yoffset * ysteps : yoffset * ysteps + ysize, :
                ] = array_tiled[t, :, :, :]
                t += 1
                yoffset += 1
            xoffset += 1

    # crop target array to target shape
    # this removes any padding to the array
    array_target = array_target[
        int(ysize * overlap) : int(ysize * overlap) + rows, int(xsize * overlap) : int(xsize * overlap) + cols, :
    ]
    return array_target.astype(dtype)


def plot_img_msk_pred(img, msk=None, pred=None, valid=None, histogram=False, n=1, title=None):
    """This function plots batches of images with respective binary masks, predictions
    and histograms.

    :param img: Images (Ndarray).
    :param msk: True masks (Ndarray).
    :param pred: Predicted masks (Ndarray).
    :param valid: Valid masks (Ndarray).
    :param histogramm: Plot histograms (Boolean).
    :param n: Number of images to plot (Integer).
    :param title: Plot title (String).

    :returns: Plot (Object).
    """
    # initiate figure layout
    cols = 1
    if msk is not None:
        cols += 1
    if pred is not None:
        cols += 1
    if histogram is True:
        cols += 1
    rows = n
    width_ratios = [1] * cols

    if img.shape[1] == 1:
        img_idx = (0, 0, 0)
    elif img.shape[1] == 2:
        img_idx = (0, 1, 1)
    else:
        img_idx = (0, 1, 2)

    fig = plt.figure(figsize=(cols, rows))

    if title:
        # add title to figure
        fig.suptitle(title)
        top_space = 0.75
    else:
        top_space = 0.95

    # https://stackoverflow.com/questions/41071947/how-to-remove-the-space-between-subplots-in-matplotlib-pyplot
    gs = gridspec.GridSpec(
        rows, cols, width_ratios=width_ratios, wspace=0.05, hspace=0.05, top=top_space, bottom=0.05, left=0.05, right=0.95
    )

    if valid is not None:
        # apply valid mask to img, msk and pred
        for i in range(rows):
            for b in range(img.shape[1]):
                img[i, b, :, :][valid[i, 0, :, :] == 0] = 0
        msk[valid == 0] = 2
        pred[valid == 0] = 2
        cm_multi = ListedColormap(["#9e9e9e", "#000080", "#000000"])
        norm = BoundaryNorm([0, 0.5, 1.5, 2], cm_multi.N)
    else:
        # plot img, msk and pred as are
        cm_multi = ListedColormap(["#ffffff", "#000000"])
        norm = BoundaryNorm([0, 0.5, 1], cm_multi.N)

    img_flat = np.ravel(img)
    img_minmax = [np.nanmin(img_flat), np.nanmax(img_flat)]

    for i in range(rows):
        # prepare rgb composite of img
        # NOTE: pytorch uses dimorder (channels, rows, cols)
        r = (
            (img[i, img_idx[0], :, :] - img[i, img_idx[0], :, :].min())
            * (1 / (img[i, img_idx[0], :, :].max() - img[i, img_idx[0], :, :].min() + 1e-8) * 255)
        ).astype(np.uint8)
        g = (
            (img[i, img_idx[1], :, :] - img[i, img_idx[1], :, :].min())
            * (1 / (img[i, img_idx[1], :, :].max() - img[i, img_idx[1], :, :].min() + 1e-8) * 255)
        ).astype(np.uint8)
        b = (
            (img[i, img_idx[2], :, :] - img[i, img_idx[2], :, :].min())
            * (1 / (img[i, img_idx[2], :, :].max() - img[i, img_idx[2], :, :].min() + 1e-8) * 255)
        ).astype(np.uint8)
        rgb = np.dstack((r, g, b))

        # plot image
        ax1 = plt.subplot(gs[i, 0])
        ax1.imshow(rgb)
        ax1.set_xticklabels([])
        ax1.set_xticks([])
        ax1.set_yticklabels([])
        ax1.set_yticks([])
        if i + 1 == rows:
            # add xaxis label on last row
            ax1.set_xlabel("Image", labelpad=7)

        if msk is not None:
            # plot mask
            ax2 = plt.subplot(gs[i, 1])
            ax2.imshow(msk[i, 0, :, :], cmap=cm_multi, norm=norm)
            ax2.set_xticklabels([])
            ax2.set_xticks([])
            ax2.set_yticklabels([])
            ax2.set_yticks([])
            if i + 1 == rows:
                # add xaxis label on last row
                ax2.set_xlabel("True Mask", labelpad=7)

        if pred is not None:
            # plot prediction
            ax3 = plt.subplot(gs[i, 2])
            ax3.imshow(pred[i, 0, :, :], cmap=cm_multi, norm=norm)
            ax3.set_xticklabels([])
            ax3.set_xticks([])
            ax3.set_yticklabels([])
            ax3.set_yticks([])
            if i + 1 == rows:
                # add xaxis label on last row
                ax3.set_xlabel("Pred Mask", labelpad=7)

        if histogram is True:
            # plot histogram
            ax3 = plt.subplot(gs[i, cols - 1])
            ax3.hist(
                np.ravel(img[i, img_idx[2], :, :]),
                bins="auto",
                range=img_minmax,
                color="#ff0000",
                histtype="stepfilled",
                alpha=0.3,
                density=True,
                label="R",
            )
            ax3.hist(
                np.ravel(img[i, img_idx[1], :, :]),
                bins="auto",
                range=img_minmax,
                color="#00ff00",
                histtype="stepfilled",
                alpha=0.3,
                density=True,
                label="G",
            )
            ax3.hist(
                np.ravel(img[i, img_idx[0], :, :]),
                bins="auto",
                range=img_minmax,
                color="#0000ff",
                histtype="stepfilled",
                alpha=0.3,
                density=True,
                label="B",
            )
            if i + 1 != rows:
                # add x-axis labels and tick on last row
                ax3.set_xticklabels([])
                ax3.set_xticks([])
            ax3.set_yticklabels([])
            ax3.set_yticks([])
