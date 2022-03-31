import argparse
import multiprocessing
import os
import random as rn
from typing import List, Tuple, Union

import cv2
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm

import config
import preprocessing.augment_op as aop
from pe_logger import PELogger

# Set seed to get reproducible augmentations
np.random.seed(1)
rn.seed(2)

logger = PELogger().get_logger()
CONFIG = config.config()


def augment(
    df_row: pd.DataFrame,
    target_w: int,
    target_h: int,
    aug_list: List[Union[List[str], str]],
    input_dir: str,
    output_dir: str,
) -> pd.DataFrame:
    """Apply data augmentation to images listed in given dataframe

    Args:
        df: dataframe consisting of image filenames and metadata to process
        target_w: pixel width to which images will be scaled
        target_h: pixel height to which image will be scaled
        aug_list: list of augmentations to apply (can be nested)
        input_dir: directory containing input images
        output_dir: output directory for augmented PNG files

    Returns:
        dataframe including rows for augmented images
    """
    new_df = pd.DataFrame(columns=df_row.columns)
    filepath = df_row["png_filename"].squeeze()
    series_instance_uid = df_row["SeriesInstanceUID"].squeeze()
    sop_instance_uid = df_row["SOPInstanceUID"].squeeze()

    im_orig = cv2.imread(os.path.join(input_dir, filepath))
    filename = os.path.basename(filepath)

    for aug in aug_list:
        df_new_row = df_row.copy()

        im = im_orig.copy()
        if isinstance(aug, list):
            for a in aug:
                im = getattr(aop, a)(im)
            im_aug = im
            aug_str = "aug_" + "_".join(aug)
        else:
            im_aug = getattr(aop, aug)(im)
            aug_str = "aug_" + aug
        im_aug = im

        if im_aug.shape[0] != target_h or im_aug.shape[1] != target_w:
            im_aug = cv2.resize(im_aug, (target_w, target_h))

        filename_aug = filename.replace(filename[-4:], "_" + aug_str + filename[-4:])
        output_filename = os.path.join(output_dir, filename_aug)

        # Update affected parameters
        df_new_row["augmented"] = 1
        df_new_row["augment_type"] = aug_str
        df_new_row["png_filename"] = filename_aug
        df_new_row["SeriesInstanceUID"] = series_instance_uid + "_" + aug_str
        df_new_row["SOPInstanceUID"] = sop_instance_uid + "_" + aug_str

        assert cv2.imwrite(output_filename, im_aug), (
            "Failed to write augmented file" + output_filename
        )

        new_df = pd.concat([new_df, df_new_row], axis=0, ignore_index=True)

    return new_df


def _set_defaults_for_new_cols(df: pd.DataFrame, target_w: int, target_h: int):
    df["rows"] = target_h
    df["columns"] = target_w
    df["augmented"] = 0
    df["augment_type"] = "none"


def _get_num_neg_per_pos_label(df: pd.DataFrame) -> int:
    counts = df["label"].value_counts()
    assert len(counts) == 2, "Dataset contains invalid labels"

    return int(counts[0] / float(counts[1]) + 0.5)


def get_aug_split_for_negative(
    df: pd.DataFrame, frac: float = 0.1
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe's negative slices to two dataframes.

    Args:
        df: dataframe with image filenames
        frac: fraction of series to augment - (i.e. not slices). Defaults to 0.1.

    Returns:
        Tuple of dataframes: first dataframe contains images not scheduled for augmentation
        and the other one in turn contains images that should be augmented.
    """
    df_neg = df.loc[df.label == 0]
    df_neg_series = df_neg.groupby("SeriesInstanceUID").first()
    df_apply_aug_series = df_neg_series.sample(frac=frac)
    df_no_aug = df_neg.loc[~df_neg["SeriesInstanceUID"].isin(df_apply_aug_series.index)]
    df_apply_aug = df_neg.loc[df_neg["SeriesInstanceUID"].isin(df_apply_aug_series.index)]

    return df_no_aug, df_apply_aug


def augment_df(
    df: pd.DataFrame,
    target_w: int,
    target_h: int,
    aug_list: List[Union[List[str], str]],
    num_augs: int,
    input_dir: str,
    output_dir: str,
) -> pd.DataFrame:
    """Augment slices listed in given dataframe in parallel by
    selecting num_augs of augmentations from aug_list.
    Returns a dataframe that includes rows for augmented images
    """
    df_aug = pd.DataFrame(columns=df.columns)

    augs_to_apply = aug_list
    if num_augs < len(aug_list):
        aug_indices = np.random.choice(np.arange(len(aug_list)), num_augs, replace=False)
        augs_to_apply = [aug_list[idx] for idx in aug_indices]

    # Process series by series with slices in parallel
    series_list = df["SeriesInstanceUID"].unique()
    for series in tqdm(series_list, unit="series"):
        df_series = df.loc[df["SeriesInstanceUID"] == series]
        df_series_aug = Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(augment)(
                df_series.iloc[row : row + 1, :],
                target_w,
                target_h,
                augs_to_apply,
                input_dir,
                output_dir,
            )
            for row in range(df_series.shape[0])
        )
        df_aug = pd.concat([df_aug, *df_series_aug], axis=0, ignore_index=True)

    return df_aug


def scale_and_copy_files_in_df(
    df: pd.DataFrame, input_dir: str, output_dir: str, target_w: int, target_h: int
):
    def scale_and_copy(filename):
        im = cv2.imread(os.path.join(input_dir, filename))
        im = cv2.resize(im, (target_w, target_h))
        cv2.imwrite(os.path.join(output_dir, filename), im)

    # Process series by series with slices in parallel
    series_list = df["SeriesInstanceUID"].unique()
    for series in tqdm(series_list, unit="series"):
        df_series = df.loc[df["SeriesInstanceUID"] == series]

        Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(scale_and_copy)(filename) for filename in df_series["png_filename"].values
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Utility to augment images based on a CSV file")

    parser.add_argument("--input-csv", dest="input_csv", required=True)
    parser.add_argument("--output-slices-csv", dest="output_slices_csv", required=True)
    parser.add_argument("--output-sequences-csv", dest="output_sequences_csv", required=True)
    parser.add_argument("--input-dir", dest="input_dir", required=True)
    parser.add_argument("--output-dir", dest="output_dir", required=True)
    args = parser.parse_args()

    aug_list = CONFIG["augmentations"]["tfxs"]
    target_w = CONFIG["augmentations"]["target_w"]
    target_h = CONFIG["augmentations"]["target_h"]

    logger.info("About to start data augmentation")
    logger.info("Augmentation consists of 3 time consuming steps")
    os.makedirs(args.output_dir, exist_ok=True)

    df = pd.read_csv(args.input_csv)
    aug_tfx_count = _get_num_neg_per_pos_label(df) - 1

    logger.info("Step 1/3 Scaling images to target size")
    scale_and_copy_files_in_df(df, args.input_dir, args.output_dir, target_w, target_h)

    _set_defaults_for_new_cols(df, target_w, target_h)

    # Compensate imbalance with augmentations, todo: try focal loss
    aug_list = aug_list[:aug_tfx_count]
    num_augs = len(aug_list)
    df_pos = df.loc[df["label"] == 1]

    logger.info("Step 2/3 Augmenting positive slices")
    df_pos_aug = augment_df(
        df_pos,
        target_w,
        target_h,
        aug_list,
        num_augs,
        args.input_dir,
        args.output_dir,
    )

    logger.info("Step 3/3 Augmenting negative slices")
    num_augs = 1
    df_neg_no_aug, df_neg_apply_aug = get_aug_split_for_negative(df, frac=0.1)

    df_neg_aug = augment_df(
        df_neg_apply_aug,
        target_w,
        target_h,
        aug_list,
        num_augs,
        args.input_dir,
        args.output_dir,
    )

    # Finally combine dataframes into one for training slices
    df_train_slices = pd.concat(
        [df_neg_no_aug, df_neg_aug, df_pos, df_pos_aug], axis=0, ignore_index=True
    )
    df_train_slices.to_csv(args.output_slices_csv, index=False)

    # Save version without augmentations for training sequences
    df.to_csv(args.output_sequences_csv, index=False)
    logger.info("Done.")
