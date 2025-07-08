import numpy as np

from scipy.signal.windows import tukey
from typing import Dict, Generator, List, Tuple


class TileGenerator:
    def __init__(
        self, array: np.ndarray, xsize: int = 256, ysize: int = 256, overlap: float = 0.1, batch_size: int = 4
    ):
        self.array = array
        self.H, self.W, self.C = array.shape
        self.xsize = xsize
        self.ysize = ysize
        self.overlap = overlap
        self.batch_size = batch_size
        self.xstep = int(xsize * (1 - overlap))
        self.ystep = int(ysize * (1 - overlap))
        self.window2d = None
        if 0 < overlap <= 0.5:
            w1d_x = tukey(self.xsize, alpha=overlap * 2)
            w1d_y = tukey(self.ysize, alpha=overlap * 2)
            self.window2d = np.expand_dims(np.outer(w1d_y, w1d_x), axis=2)  # (H, W, 1)

    def tile_array(self) -> Generator[Tuple[np.ndarray, List[Tuple[int, int]]], None, None]:
        batch, positions = [], []
        for y in range(0, self.H, self.ystep):
            for x in range(0, self.W, self.xstep):
                tile = self._get_tile(y, x)
                batch.append(tile)
                positions.append((y, x))
                if len(batch) == self.batch_size:
                    yield np.stack(batch), positions
                    batch, positions = [], []
        if batch:
            yield np.stack(batch), positions

    def _get_tile(self, y: int, x: int) -> np.ndarray:
        y_end = min(y + self.ysize, self.H)
        x_end = min(x + self.xsize, self.W)
        tile = self.array[y:y_end, x:x_end, :]
        pad_y = self.ysize - tile.shape[0]
        pad_x = self.xsize - tile.shape[1]
        if pad_y > 0 or pad_x > 0:
            tile = np.pad(tile, ((0, pad_y), (0, pad_x), (0, 0)), mode="symmetric")
        return tile

    def untile_array(
        self,
        batches: Generator[np.ndarray, None, None],
        positions: Generator[List[Tuple[int, int]], None, None],
        smooth_blending: bool = False,
    ) -> np.ndarray:
        C = batches[0].shape[-1]
        result = np.zeros((self.H, self.W, C), dtype=np.float32)
        count = np.zeros((self.H, self.W, C), dtype=np.float32)
        for tile, position in zip(batches, positions):
            for i, (y, x) in enumerate(position):
                pred = tile[i]
                if smooth_blending and self.window2d is not None:
                    pred = pred * self.window2d  # shape (H, W, C) broadcast
                h = min(pred.shape[0], self.H - y)
                w = min(pred.shape[1], self.W - x)
                result[y : y + h, x : x + w, :] += pred[:h, :w, :]
                count[y : y + h, x : x + w, :] += (
                    self.window2d[:h, :w, :] if smooth_blending and self.window2d is not None else 1.0
                )
        return (result / np.maximum(count, 1e-6)).astype(self.array.dtype)


def reclassify(array: np.ndarray, class_dict: Dict[str, List[int]]) -> np.ndarray:
    array_rec = np.zeros((array.shape[0], array.shape[1], 1), dtype=np.uint8)
    for i in range(len(class_dict["reclass_value_from"])):
        array_rec[array == class_dict["reclass_value_from"][i]] = class_dict["reclass_value_to"][i]

    return array_rec.astype(np.uint8)


def cohen_kappa_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape != y_pred.shape:
        raise TypeError("y_true.shape must match y_pred.shape")

    po = (y_true == y_pred).astype(np.float32).mean()
    classes = sorted(set(list(np.concatenate((y_true, y_pred), axis=0))))

    mp = {}
    for i, c in enumerate(classes):
        mp[c] = i
    k = len(mp)

    sa = np.zeros(shape=(k,), dtype=np.int32)
    sb = np.zeros(shape=(k,), dtype=np.int32)
    n = y_true.shape[0]
    for x, y in zip(list(y_true), list(y_pred)):
        sa[mp[x]] += 1
        sb[mp[y]] += 1

    pe = 0
    for i in range(k):
        pe += (sa[i] / n) * (sb[i] / n)

    kappa = (po - pe) / (1.0 - pe)

    return kappa
