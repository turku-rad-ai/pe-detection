import os
from typing import Callable, List, Tuple

import cv2
import numpy as np
import pandas as pd
from tensorflow import keras


class SliceDataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        df: pd.DataFrame,
        image_dir: str,
        filename_col: str,
        label_col: str,
        batch_size: int,
        dim: Tuple[int, int],
        num_channels: int,
        preprocess_fn: Callable,
        shuffle=True,
    ):
        self.df = df.copy()
        self.filenames = self.df[filename_col].unique()
        self.image_dir = image_dir
        self.dim = dim
        self.batch_size = batch_size
        self.label_col = label_col
        self.filename_col = filename_col
        self.num_channels = num_channels
        self.preprocess_fn = preprocess_fn
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self) -> int:
        """Return number of batches per epoch"""
        return int(np.ceil(self.df.shape[0] / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        offset = index * self.batch_size
        indexes = self.indexes[offset : min(offset + self.batch_size, self.df.shape[0])]

        # Find list of UIDs
        UIDs = [self.filenames[k] for k in indexes]

        # Generate data
        X, y = self._prepare_batch(UIDs)

        return X, y

    def on_epoch_end(self):
        self.indexes = np.arange(self.df.shape[0])
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _prepare_batch(self, UIDs: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        batch_size = len(UIDs)
        X = np.zeros((batch_size, *self.dim, self.num_channels), dtype=np.float32)
        y = np.zeros((batch_size, 1), dtype=np.int32)

        # Generate data
        for i, uid in enumerate(UIDs):
            df_im = self.df.loc[self.df[self.filename_col] == uid]
            fn = df_im[self.filename_col].iloc[0]
            im = cv2.imread(os.path.join(self.image_dir, fn))[:, :, 0 : self.num_channels]

            X[i, :, :, :] = im[:, :, :]
            y[i, 0] = df_im[self.label_col].iloc[0]

        X = self.preprocess_fn(X)

        return X, y

    def get_df(self) -> pd.DataFrame:
        return self.df
