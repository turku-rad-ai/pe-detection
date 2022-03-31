import argparse
import glob
import os
import sys
from typing import Any, Dict, List

import cv2
import numpy as np
import pandas as pd
import pydicom
from keras.applications.inception_resnet_v2 import preprocess_input
from keras.models import Model, load_model
from tqdm import tqdm

import config
from pe_logger import PELogger
from preprocessing.dcm_to_png import load_image_from_dcm

logger = PELogger().get_logger()
CONFIG = config.config()
ENCODING_DIM = CONFIG["model"]["encoding_dim"]
ENCODING_LAYER_IDX = CONFIG["model"]["encoding_layer_idx"]
NUM_SLICES = CONFIG["model"]["sequence_num_slices"]
IMAGE_W = CONFIG["augmentations"]["target_w"]
IMAGE_H = CONFIG["augmentations"]["target_h"]

PREPROCESSED_DIR = "./generated/eval_pngs"  # folder where the preprocessed images are saved

SOP_CLASS_UID_CT_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.2"
SOP_CLASS_UID_SECONDARY_CAPTURE_IMAGE_STORAGE = "1.2.840.10008.5.1.4.1.1.7"
MAX_WINDOW_WIDTH = 700


class DetectionWorker:
    def __init__(
        self,
        input_dir: str,
        window: int,
        level: int,
        num_channels: int,
        slice_model_path: str,
        sequence_model_path: str,
        output_csv: str,
        save_pngs: bool = False,
    ):
        self.window = window
        self.level = level
        self.num_channels = num_channels
        self.save_pngs = save_pngs

        if self.save_pngs:
            os.makedirs(PREPROCESSED_DIR, exist_ok=True)

        # Load Keras models
        self.slice_model = load_model(slice_model_path)
        self.encoding_model = Model(
            inputs=self.slice_model.input,
            outputs=[self.slice_model.layers[ENCODING_LAYER_IDX].output],
        )  # Get encoding using model without classifier

        self.sequence_model = load_model(sequence_model_path)

        # Get dataframe with filenames of DICOM files in the folder
        self.df_images = self._search_for_new_files(input_dir)
        self.df_images["pred_slice"] = -1.0
        self.df_images["pred_stack"] = -1.0

        # Run detection and save results to a CSV file
        self._detect_and_report_images(self.df_images)
        self.df_images.to_csv(output_csv, index=False)

    def _search_for_new_files(self, input_dir: str) -> pd.DataFrame:
        dirs_and_files = glob.glob(os.path.join(input_dir, "**"), recursive=True)
        file_list = []

        for file in dirs_and_files:
            if file.lower().endswith(".dcm"):
                file_list.append(file)

        data: Dict[str, Any] = {}

        # Initialize preferred set of DICOM tags to a dataframe
        hparams = self._header_params()
        for p in hparams:
            data[p] = []

        # Initialize fields for filename and predictions
        data["filename"] = []
        data["pred_slice"] = []
        data["pred_stack"] = []

        for f in file_list:
            ds = pydicom.dcmread(f, stop_before_pixels=True)

            if not self._is_file_valid_for_detection(ds):
                continue

            data["filename"].append(f)
            data["pred_slice"].append(-1.0)
            data["pred_stack"].append(-1.0)

            # Fill in some values from DICOM header to the dataframe
            for p in hparams:
                if p in ds:
                    de = ds.data_element(p)
                    if de is not None:
                        data[p].append(de.value)
                else:
                    data[p].append(None)

        df_images = pd.DataFrame(data)

        return df_images

    def _img_orient_pat_dist(
        self, test_orient: pydicom.multival.MultiValue, ref_orient: List[float]
    ) -> float:
        """Combine x and y vectors into one and return L2-norm of test and ref vector difference"""
        try:
            ref_vec = np.array(list(map(float, ref_orient)))
            test_vec = np.array(list(map(float, test_orient)))

            dist = np.linalg.norm(ref_vec - test_vec)
        except Exception:
            return sys.float_info.max

        return float(dist)

    def _is_file_valid_for_detection(self, ds: pydicom.dataset.FileDataset) -> bool:
        """Check whether a DICOM image matches to how models have been trained.

        Args:
            ds: DICOM header information

        Validity is based on following criteria:
            - Slice must have thickness close to 3 mm
            - Slice must have suitable SOPClassUID
            - Slice orientation based on ImageOrientationPatient - tag,
              needs to indicate an axial slice
            - Windowing that matches correct PE protocol

        Returns:
            True if series is valid for detection
        """

        ref_pat_orient = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0]
        if (
            "SliceThickness" in ds
            and ds.SliceThickness > 2.95
            and ds.SliceThickness < 3.05
            and (
                "SOPClassUID" in ds
                and ds.SOPClassUID == SOP_CLASS_UID_CT_IMAGE_STORAGE
                or ds.SOPClassUID == SOP_CLASS_UID_SECONDARY_CAPTURE_IMAGE_STORAGE
            )
            and self._img_orient_pat_dist(ds.ImageOrientationPatient, ref_pat_orient) < 0.05
            and ("WindowWidth" in ds)
            and ds.WindowWidth < MAX_WINDOW_WIDTH
        ):
            return True
        return False

    def _detect_and_report_images(self, df: pd.DataFrame):
        series_uids = df["SeriesInstanceUID"].unique()

        for series_uid in tqdm(series_uids, unit="series"):
            X = self._get_tensor_for_series(df, series_uid)
            encodings = self.encoding_model.predict(X)

            encodings = encodings.reshape((1, NUM_SLICES, ENCODING_DIM))
            prediction = self.sequence_model.predict(encodings)
            slice_preds = prediction[0].flatten()
            stack_preds = prediction[1].flatten()

            for i in range(NUM_SLICES):
                if (
                    df.loc[
                        ((df["SeriesInstanceUID"] == series_uid) & (df["InstanceNumber"] == i))
                    ].shape[0]
                    > 0
                ):
                    df.loc[
                        ((df["SeriesInstanceUID"] == series_uid) & (df["InstanceNumber"] == i)),
                        "pred_slice",
                    ] = slice_preds[i]
                    df.loc[
                        ((df["SeriesInstanceUID"] == series_uid) & (df["InstanceNumber"] == i)),
                        "pred_stack",
                    ] = stack_preds[0]

    def _get_tensor_for_series(self, df_images: pd.DataFrame, series_uid: str) -> np.ndarray:
        """Return a tensor of preprocessed slice images to be used with the encoding model.
        The size of the first dimension is matched with the size of LSTM model input.

        Args:
            df_images: dataframe containing filenames and required metadata such as instance number
            series_uid: identifier of the series/image for which tensor is created

        Returns:
            preprocessed tensor
        """
        df_series = df_images.loc[df_images["SeriesInstanceUID"] == series_uid]
        df_series = df_series.sort_values(
            by=["InstanceNumber"]
        )  # Make sure ordering of slices is correct

        # Initialization
        X = np.zeros((NUM_SLICES, IMAGE_H, IMAGE_W, self.num_channels), dtype=np.float32)

        # Generate data
        for slice_idx, fn in enumerate(df_series["filename"].values):
            if slice_idx >= NUM_SLICES:
                break

            windowed = load_image_from_dcm(fn, self.window, self.level, do_segmentation=True)
            windowed = cv2.resize(windowed, (IMAGE_W, IMAGE_H))

            if self.save_pngs:
                png_filename = os.path.join(PREPROCESSED_DIR, f"{series_uid}_{slice_idx:03d}.png")
                assert cv2.imwrite(
                    png_filename, windowed
                ), f"Failed to save PNG file {png_filename}"

            windowed = preprocess_input(windowed)

            X[slice_idx, :, :, 0 : self.num_channels] = windowed[:, :, 0 : self.num_channels]

        return X

    def _header_params(self) -> List[str]:
        """Return the DICOM tags that we want to need for detection and want to report"""
        params = ["SeriesInstanceUID", "InstanceNumber"]

        return params


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run PE detection models for 3mm axial slices in DICOM format"
    )
    parser.add_argument(
        "--input-dir", dest="input_dir", required=True, help="Folder containing DICOM files to test"
    )
    parser.add_argument("--window", dest="window", required=True, type=int)
    parser.add_argument("--level", dest="level", required=True, type=int)
    parser.add_argument(
        "--num-channels",
        required=True,
        type=int,
        help="Number of channels to be used for input of the slice model",
    )
    parser.add_argument("--slice-model", dest="slice_model_path", required=True)
    parser.add_argument("--sequence-model", dest="sequence_model_path", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument(
        "--save-pngs",
        action="store_true",
        help="Flag whether to store pre-processed slices as PNGs",
    )
    args = parser.parse_args()

    logger.info("Starting detection for PE, this might take a while..")

    # fmt: off
    DetectionWorker(
        args.input_dir,
        args.window,
        args.level,
        args.num_channels,
        args.slice_model_path,
        args.sequence_model_path,
        args.output_csv,
        args.save_pngs
    )
    # fmt: on

    logger.info("Done.")
