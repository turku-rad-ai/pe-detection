import argparse
import multiprocessing
import os
from typing import Dict, NamedTuple

import cv2
import numpy as np
import pandas as pd
import pydicom
from joblib import Parallel, delayed
from tqdm import tqdm

from pe_logger import PELogger
from preprocessing.segmentation import apply_mask, get_lungs_and_heart_mask

logger = PELogger().get_logger()

# Use wide enough windowing for segmentation so that lungs are visible in images
WINDOW_SEG = 1400
LEVEL_SEG = -500


class PNGInfo(NamedTuple):
    filename: str
    instance_number: int
    valid: bool


def get_windowed_version(pixels: np.ndarray, window: int, level: int) -> np.ndarray:
    """Apply windowing to image to optimize pixel values for further processing"""
    assert pixels.dtype == np.float32, "Windowing only supported for float images"

    return np.piecewise(
        pixels,
        [
            pixels <= (level - 0.5 - (window - 1) / 2),
            pixels > (level - 0.5 + (window - 1) / 2),
        ],
        [
            0,
            255,
            lambda data: ((data - (level - 0.5)) / (window - 1) + 0.5) * (255 - 0),
        ],
    )


def load_image_from_dcm(
    dcm_fullpath: str, window: int, level: int, do_segmentation: bool
) -> np.ndarray:
    """Load image from a DICOM file and return it as an 8-bit numpy ndarray"""

    ds = pydicom.dcmread(dcm_fullpath, stop_before_pixels=False)
    assert ds is not None, f"Could not find DICOM file {dcm_fullpath}"

    pixels = ds.pixel_array
    assert pixels is not None, f"DICOM file {dcm_fullpath} does not contain pixel data"

    pixels = pixels.astype(np.float32)

    # Convert data to Hounsfield units
    pixels *= ds.RescaleSlope
    pixels += int(ds.RescaleIntercept)

    windowed = get_windowed_version(pixels, window, level)
    im_windowed = windowed.astype(np.uint8)
    im_windowed = cv2.cvtColor(im_windowed, cv2.COLOR_GRAY2RGB)

    if do_segmentation:
        windowed_for_seg = get_windowed_version(pixels, WINDOW_SEG, LEVEL_SEG)
        im_windowed_for_seg = windowed_for_seg.astype(np.uint8)
        mask = get_lungs_and_heart_mask(im_windowed_for_seg)
        im_windowed = apply_mask(im_windowed, mask)

    return im_windowed


def to_png(
    dcm_fullpath: str,
    window: int,
    level: int,
    label: int,
    dataset_label: int,
    out_dir: str,
    sop_instance_uid: str,
    do_segmentation: bool,
) -> PNGInfo:
    im_windowed = load_image_from_dcm(dcm_fullpath, window, level, do_segmentation)

    ds = pydicom.dcmread(dcm_fullpath, stop_before_pixels=True)
    instance_number = f"{ds.InstanceNumber:03d}"

    # Positive slices are not allowed to be empty
    valid = label == 0 or np.max(im_windowed) > 0

    png_fn = f"I{label:01d}_D{dataset_label:01d}_S{instance_number}_V{int(valid)}_"
    png_fn += sop_instance_uid.replace(".", "_") + ".png"
    png_fp = os.path.join(out_dir, png_fn)

    assert cv2.imwrite(png_fp, im_windowed), f"Failed to save PNG file {png_fp}"

    return PNGInfo(png_fn, ds.InstanceNumber, valid)


def map_to_png_params(
    study_dir: str,
    df_row: pd.DataFrame,
    window: int,
    level: int,
    do_segmentation: bool,
    out_dir: str,
) -> Dict:
    """Convenience method to generate parameters for to_png()"""
    data = {
        "dcm_fullpath": os.path.join(
            study_dir,
            df_row["StudyInstanceUID"],
            df_row["SeriesInstanceUID"],
            df_row["dcm_filename"],
        ),
        "window": window,
        "level": level,
        "label": df_row["label"],
        "dataset_label": df_row["dataset_label"],
        "out_dir": out_dir,
        "sop_instance_uid": df_row["SOPInstanceUID"],
        "do_segmentation": do_segmentation,
    }

    return data


def process_df(
    df: pd.DataFrame,
    study_dir: str,
    png_dir: str,
    window: int,
    level: int,
    do_segmentation: bool,
    output_csv: str,
):
    """Run parallel processing jobs to convert DICOM images specified in dataframe to png"""
    png_filenames = []
    instance_numbers = []
    valid = []

    splits = np.array_split(np.arange(df.shape[0]), max(df.shape[0] // 100, 1))

    for split in tqdm(splits, unit="chunk"):
        split_results = Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(to_png)(
                **map_to_png_params(study_dir, df_row, window, level, do_segmentation, png_dir)
            )
            for _, df_row in df.iloc[split].iterrows()
        )

        png_filenames.extend([r.filename for r in split_results])
        instance_numbers.extend([r.instance_number for r in split_results])
        valid.extend([r.valid for r in split_results])

    df.insert(0, "png_filename", png_filenames)
    df.insert(0, "InstanceNumber", instance_numbers)  # (0020, 0013)
    df.insert(0, "valid", valid)
    df.to_csv(output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Utility to convert DICOM images to PNGs and create a corresponding CSV file"
    )

    parser.add_argument("--study-dir", required=True)
    parser.add_argument("--png-dir", required=True)
    parser.add_argument("--window", required=True, type=int)
    parser.add_argument("--level", required=True, type=int)
    parser.add_argument("--segmentation", dest="segmentation", action="store_true")
    parser.add_argument("--no-segmentation", dest="segmentation", action="store_false")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.set_defaults(segmentation=True)

    args = parser.parse_args()

    os.makedirs(args.png_dir, exist_ok=True)

    if not os.path.exists(args.input_csv):
        logger.error(f"Could not find CSV file {args.input_csv}")
    else:
        df = pd.read_csv(args.input_csv)

        logger.info("Starting DICOM to PNG conversion - this will take a while.")
        process_df(
            df,
            args.study_dir,
            args.png_dir,
            args.window,
            args.level,
            args.segmentation,
            args.output_csv,
        )
        logger.info("Done.")
