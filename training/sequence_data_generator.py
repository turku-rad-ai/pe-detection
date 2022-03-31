from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from tensorflow import keras


class SequenceDataGenerator(keras.utils.Sequence):
    def __init__(
        self,
        df: pd.DataFrame,
        encoding_dict: Dict[str, np.ndarray],
        filename_col: str,
        label_col: str,
        dataset_label_col: str,
        batch_size: int,
        encoding_dim: int,
        num_slices: int = 96,
        shuffle: bool = True,
    ):
        self.df = df.copy()
        self.series_instance_uids = self.df["SeriesInstanceUID"].unique()
        self.count_no_aug_series = len(self.series_instance_uids)
        self.encoding_dict = encoding_dict
        self.filename_col = filename_col
        self.label_col = label_col
        self.dataset_label_col = dataset_label_col
        self.batch_size = batch_size
        self.encoding_dim = encoding_dim
        self.num_slices = num_slices
        self.shuffle = shuffle

        self.on_epoch_end()

    def __len__(self) -> int:
        return int(np.ceil(self.count_no_aug_series / self.batch_size))

    def __getitem__(self, index: int) -> Tuple[np.ndarray, List[np.ndarray]]:
        offset = index * self.batch_size
        indexes = self.indexes[offset : min(offset + self.batch_size, self.df.shape[0])]

        UIDs = [self.series_instance_uids[k] for k in indexes]

        X, ys = self._prepare_batch(UIDs)

        return X, ys

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        self.indexes = np.arange(len(self.series_instance_uids))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _prepare_batch(self, UIDs: List[str]) -> Tuple[np.ndarray, List[np.ndarray]]:
        batch_size = len(UIDs)
        X = np.zeros((batch_size, self.num_slices, self.encoding_dim), dtype=np.float32)
        y = np.zeros((batch_size, self.num_slices, 1), dtype=np.int32)
        y_stack = np.zeros((batch_size, 1), dtype=np.int32)

        for i, uid in enumerate(UIDs):
            df_im = self.df.loc[self.df["SeriesInstanceUID"] == uid]
            df_im = df_im.sort_values(by=["InstanceNumber"])

            for slice_idx, fn in enumerate(df_im[self.filename_col].values[: self.num_slices]):
                X[i, slice_idx, :] = self.encoding_dict[fn]

                # Store class
                label = df_im[self.label_col].values[slice_idx]
                dataset_label = df_im[self.dataset_label_col].values[slice_idx]
                y[i, slice_idx, 0] = label
                y_stack[i, 0] = dataset_label

        return X, [y, y_stack]
