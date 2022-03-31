import argparse
import os
import random as rn
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow import keras

import config
import training.inception_resnet_v2_gray  # Supress TF lambda layer warning
from pe_logger import PELogger
from training.slice_data_generator import SliceDataGenerator

# Set seeding based on Keras documentation
np.random.seed(1)
rn.seed(2)
tf.random.set_seed(3)

logger = PELogger().get_logger()

CONFIG = config.config()
ENCODING_DIR = CONFIG["model"]["encoding_dir"]
ENCODING_CSV = CONFIG["model"]["encoding_csv"]
IMAGE_W = CONFIG["augmentations"]["target_w"]
IMAGE_H = CONFIG["augmentations"]["target_h"]
ENCODING_DIM = CONFIG["model"]["encoding_dim"]
ENCODING_LAYER_IDX = CONFIG["model"]["encoding_layer_idx"]
MODEL_DIR = CONFIG["model"]["model_dir"]
FILENAME_COL = "png_filename"
LABEL_COL = "label"
BATCH_SIZE = 48


def encode(
    df_no_aug: pd.DataFrame, image_dir: str, model_prefix: str, num_channels: int, fold: int
):
    full_model = keras.models.load_model(
        os.path.join(MODEL_DIR, f"{model_prefix}_fold_{fold:02d}.h5")
    )

    encoding_model = keras.models.Model(
        inputs=full_model.input, outputs=[full_model.layers[ENCODING_LAYER_IDX].output]
    )

    generator = SliceDataGenerator(
        df_no_aug,
        image_dir,
        FILENAME_COL,
        LABEL_COL,
        BATCH_SIZE,
        (IMAGE_H, IMAGE_W),
        num_channels,
        preprocess_input,
        shuffle=False,
    )

    encodings = np.zeros((df_no_aug.shape[0], ENCODING_DIM))
    predict = encoding_model.predict(
        generator, steps=np.ceil(df_no_aug.shape[0] / BATCH_SIZE), verbose=1
    )
    encodings[:, :] = predict[: df_no_aug.shape[0]]

    np.save(os.path.join(ENCODING_DIR, model_prefix + f"_encodings_fold_{fold:02d}.npy"), encodings)


def store_encoding_csv(df: pd.DataFrame):
    os.makedirs(ENCODING_DIR, exist_ok=True)
    df["png_filename"].to_csv(os.path.join(ENCODING_DIR, ENCODING_CSV), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--model-prefix", required=True)
    parser.add_argument("--num-channels", type=int, required=True)
    parser.add_argument("--num-folds", type=int, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        logger.error(f"Could not find CSV file {args.input_csv}")
        sys.exit(os.EX_NOINPUT)

    df = pd.read_csv(args.input_csv)
    df_no_aug = df.loc[df["augmented"] == 0]  # Only get encodings for non-augmented images

    store_encoding_csv(df)

    logger.info(f"About to calculate encodings for {args.num_folds} folds")
    for fold in range(args.num_folds):
        logger.info(f"Processing fold: {fold + 1} / {args.num_folds}")
        encode(df_no_aug, args.image_dir, args.model_prefix, args.num_channels, fold)
