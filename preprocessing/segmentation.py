import argparse
import glob
import multiprocessing
import os

import cv2
import numpy as np
from joblib import Parallel, delayed
from scipy import ndimage
from tqdm import tqdm

"""Lungs and heart segmentation in axial slices loosely based on paper:
A Comprehensive Framework for Automatic Detection of Pulmonary Nodules in Lung CT Images
by Mehdi Alilou,1, Vassili Kovalev, Eduard Snezhko and Vahid Taimouri

Axial slices are expected to be 512 x 512 pixels in size
"""


MAX_8BIT = 255
BODY_MIN_AREA = 50000
BODY_MAX_AREA = 300000
BODY_COM_MIN_X = 190
BODY_COM_MAX_X = 310
BODY_COM_MIN_Y = 190
BODY_COM_MAX_Y = 360
EXCLUDE_BLOB_SHAPE_RATIO = 0.35


def _invert(im: np.ndarray) -> np.ndarray:
    return MAX_8BIT - im


def _body_mask(im_bin: np.ndarray) -> np.ndarray:
    """Get a binary mask for the body"""
    retval, labels = cv2.connectedComponents(im_bin.astype(np.uint8))

    body_mask = None
    im_tmp = np.zeros_like(im_bin)

    # Find blob that matches body
    for b in range(1, retval):
        im_tmp[:, :] = 0

        blob_pixel_indices = np.where(labels == b)
        pixel_count = len(blob_pixel_indices[0])

        im_tmp[blob_pixel_indices] = MAX_8BIT
        com = ndimage.measurements.center_of_mass(im_tmp)

        if (
            pixel_count > BODY_MIN_AREA
            and pixel_count < BODY_MAX_AREA
            and com[0] > BODY_COM_MIN_Y
            and com[0] < BODY_COM_MAX_Y
            and com[1] > BODY_COM_MIN_X
            and com[1] < BODY_COM_MAX_X
        ):
            body_mask = im_tmp.copy()
            break

    if body_mask is None:
        # Did not find a body in the image
        return np.zeros_like(im_bin)

    # Extract outer contour so that we can fill possible holes
    contours, _ = cv2.findContours(body_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_lengths = [len(c) for c in contours]
    max_c_idx = np.argmax(np.array(contour_lengths))

    body_mask[:, :] = 0
    cv2.drawContours(body_mask, [contours[max_c_idx]], -1, [MAX_8BIT] * 3, cv2.FILLED)

    return body_mask


def get_lungs_and_heart_mask(im_gray: np.ndarray) -> np.ndarray:
    """Lungs and heart segmentation

    Args:
        im_gray: image, expects shape (512, 512, 1) and dtype np.uint8

    Returns:
        get binary mask for lungs and heart
    """
    assert im_gray.shape == (512, 512, 1)

    kernel = np.ones((3, 3), np.uint8)

    # Binarize grayscale image using OTSU Thresholding
    _, im_bin = cv2.threshold(im_gray, 0, MAX_8BIT, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    im_init_lung_mask = _invert(im_bin)
    im_secondary_lung_mask = cv2.bitwise_and(im_init_lung_mask, _body_mask(im_bin))

    # Remove salt and pepper noise
    im_secondary_lung_mask = cv2.medianBlur(im_secondary_lung_mask, ksize=5)

    # Close cavities
    im_closed_lung_mask = cv2.morphologyEx(
        im_secondary_lung_mask, cv2.MORPH_CLOSE, kernel, iterations=8
    )

    segmented_lungs = cv2.bitwise_and(im_gray, im_closed_lung_mask)
    _, im_lungs_thr = cv2.threshold(segmented_lungs, 1, 255, cv2.THRESH_BINARY)

    # Remove small blobs
    retval, labels, stats, _ = cv2.connectedComponentsWithStats(
        im_lungs_thr.astype(np.uint8), connectivity=8
    )

    im_tmp = np.zeros_like(im_lungs_thr)
    h, w = im_gray.shape[:2]

    for b in range(1, retval):
        im_tmp[:, :] = 0

        blob_pixel_indices = np.where(labels == b)
        im_tmp[blob_pixel_indices] = MAX_8BIT

        # Remove too small blobs
        if len(blob_pixel_indices[0]) < (0.001 * w * h):
            im_lungs_thr[blob_pixel_indices] = 0

        # Remove horizontally dominant blobs
        if (
            stats[b][cv2.CC_STAT_HEIGHT] > 0
            and stats[b][cv2.CC_STAT_WIDTH] / stats[b][cv2.CC_STAT_HEIGHT]
        ) < EXCLUDE_BLOB_SHAPE_RATIO:
            im_lungs_thr[blob_pixel_indices] = 0

    # Remove thin structures
    im_lungs_thr = cv2.morphologyEx(im_lungs_thr, cv2.MORPH_ERODE, kernel, iterations=1)
    im_lungs_thr = cv2.morphologyEx(im_lungs_thr, cv2.MORPH_DILATE, kernel, iterations=3)

    # Finding contours for the thresholded image
    contours, _ = cv2.findContours(im_lungs_thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        black = np.zeros_like(im_bin)
        return black

    # create array for convex hull points
    ccs = np.empty((0, 1, 2), dtype=np.int32)
    for i in range(len(contours)):
        ccs = np.concatenate((ccs, np.array(contours[i], dtype=np.int32)), axis=0)

    hull = [cv2.convexHull(ccs, False)]

    lungs_and_heart_mask = np.zeros_like(im_gray)
    cv2.drawContours(lungs_and_heart_mask, hull, -1, [MAX_8BIT] * 3, thickness=cv2.FILLED)

    return lungs_and_heart_mask


def apply_mask(im: np.ndarray, im_mask: np.ndarray) -> np.ndarray:
    if im.ndim == 3 and im.shape[2] > 1:
        segmentation_mask = cv2.cvtColor(im_mask, cv2.COLOR_GRAY2RGB)
    else:
        segmentation_mask = im_mask

    segmented = cv2.bitwise_and(im, segmentation_mask)

    return segmented


def segment(im: np.ndarray) -> np.ndarray:
    segmentation_mask = get_lungs_and_heart_mask(im)
    segmented_image = apply_mask(im, segmentation_mask)
    return segmented_image


def segment_file(img_path: str, out_filename: str):
    im = cv2.imread(img_path)
    segmentation_mask = get_lungs_and_heart_mask(im)
    im_seg = apply_mask(im, segmentation_mask)
    cv2.imwrite(out_filename, im_seg)


def segment_folder(input_dir: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    png_files = sorted(glob.glob(os.path.join(input_dir, "*.png")))

    for file_split in tqdm(np.array_split(png_files, 1000), unit="chunk"):
        seg_filenames = [os.path.join(output_dir, os.path.split(f)[1]) for f in file_split]
        Parallel(n_jobs=multiprocessing.cpu_count())(
            delayed(segment_file)(in_fn, seg_fn) for in_fn, seg_fn in zip(file_split, seg_filenames)
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Segment heart and lungs from chest CT image slice"
    )
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", required=True)

    args = parser.parse_args()

    segment_folder(args.input_dir, args.output_dir)
