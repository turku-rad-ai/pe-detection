import argparse
import os
import random as rn
import sys
from typing import Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from keras.applications.inception_resnet_v2 import preprocess_input
from tensorflow import keras

import config
from pe_logger import PELogger
from plots.plot_results import (save_accuracy_plot, save_loss_plot,
                                save_pr_curve, save_roc_curve)
from training.inception_resnet_v2_gray import InceptionResNetV2Gray
from training.slice_data_generator import SliceDataGenerator

# Set seeding based on Keras documentation
np.random.seed(1)
rn.seed(2)
tf.random.set_seed(3)

logger = PELogger().get_logger()

CONFIG = config.config()
MODEL_DIR = CONFIG["model"]["model_dir"]
PLOT_DIR = CONFIG["model"]["plot_dir"]
IMAGE_W = CONFIG["augmentations"]["target_w"]
IMAGE_H = CONFIG["augmentations"]["target_h"]
ENCODING_DIM = CONFIG["model"]["encoding_dim"]
FILENAME_COL = "png_filename"
LABEL_COL = "label"
FOLD_COL = "pat_fold"
EPOCHS = 8
BATCH_SIZE = 48
PRETRAINED_WEIGHTS = "./pretrained/InceptionResNetV2_NIH15_Px256.h5"


def get_model(num_channels: int) -> keras.models.Model:
    """Get grayscale or RGB Inception ResNet V2 model with custom classifier"""
    inputs = keras.Input(shape=(IMAGE_H, IMAGE_W, num_channels))

    if num_channels > 1:
        inception = keras.applications.InceptionResNetV2(
            input_shape=(IMAGE_H, IMAGE_W, num_channels), weights="imagenet", include_top=False
        )
    else:
        inception = InceptionResNetV2Gray(
            input_shape=(IMAGE_H, IMAGE_W, num_channels),
            include_top=False,
            weights=PRETRAINED_WEIGHTS,
        )

    inception.trainable = False

    x = inception(inputs, training=False)  # preserve BN
    x = keras.layers.GlobalAveragePooling2D()(inception.output)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(ENCODING_DIM)(x)
    x = keras.layers.Dense(1, activation="sigmoid")(x)
    model = keras.models.Model(inputs=inception.input, outputs=x)

    return model


def get_generators(
    df: pd.DataFrame, image_dir: str, num_channels: int, fold: int
) -> Tuple[SliceDataGenerator, SliceDataGenerator]:
    """Get training and validation generators for training"""
    df_train = df.loc[(df[FOLD_COL] != fold) & df["valid"]]
    df_test = df.loc[(df[FOLD_COL] == fold) & (df["augmented"] == 0) & df["valid"]]

    train_generator = SliceDataGenerator(
        df_train,
        image_dir,
        FILENAME_COL,
        LABEL_COL,
        BATCH_SIZE,
        (IMAGE_H, IMAGE_W),
        num_channels,
        preprocess_input,
        shuffle=True,
    )

    valid_generator = SliceDataGenerator(
        df_test,
        image_dir,
        FILENAME_COL,
        LABEL_COL,
        BATCH_SIZE,
        (IMAGE_H, IMAGE_W),
        num_channels,
        preprocess_input,
        shuffle=False,
    )

    return train_generator, valid_generator


def train(
    df: pd.DataFrame,
    image_dir: str,
    model_prefix: str,
    num_channels: int,
    pre_train_top: bool,
    fold: int,
):
    """Train a slice based (2D) binary classification model using Inception ResNet V2 backbone.
    Best model for given fold will be saved based validation loss.

    Args:
        df: dataframe object that contains list of images with labels and fold info
        image_dir: directory from which training images are loaded from
        model_prefix: prefix to use when storing trained models
        num_channels: number of input channels to use, 1 = NIH, 2 = ImageNet (RGB)
        pre_train_top: flag whether to first train a the classifier part
        fold: number indicating cross-validation fold
    """
    model = get_model(num_channels)

    train_generator, valid_generator = get_generators(df, image_dir, num_channels, fold)

    opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, decay=0.01)

    model_fn = os.path.join(MODEL_DIR, f"{model_prefix}_fold_{fold:02d}.h5")
    training_histories = []
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOT_DIR, exist_ok=True)

    mc = keras.callbacks.ModelCheckpoint(
        model_fn,
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    # With small datasets, train first 1 epoch with backbone frozen
    if pre_train_top:
        logger.info(f"Pre-training classifier first")
        model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
        if fold == 0:
            model.summary()

        hist_frozen = model.fit(
            train_generator,
            steps_per_epoch=len(train_generator),
            epochs=1,
            shuffle=False,  # Shuffling done in generator
            validation_data=valid_generator,
            validation_steps=len(valid_generator),
            callbacks=[mc],
            verbose=True,
        )
        training_histories.append(pd.DataFrame(hist_frozen.history))

        model = keras.models.load_model(model_fn)  # Load best if we ran several epochs
        opt = keras.optimizers.Adam(learning_rate=0.00001, beta_1=0.9, beta_2=0.999, decay=0.003)
        logger.info(f"Starting fine-tuning the model")

    # Set the whole model trainable
    for layer in model.layers:
        layer.trainable = True

    # Compile after modifying trainable layers
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])
    if fold == 0:
        model.summary()

    # Train whole model
    hist_full = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=EPOCHS,
        shuffle=False,  # Shuffling done in generator
        validation_data=valid_generator,
        validation_steps=len(valid_generator),
        callbacks=[mc],
    )

    training_histories.append(pd.DataFrame(hist_full.history))

    df_hist = pd.concat(training_histories, axis=0, ignore_index=True, sort=False)
    df_hist.to_csv(os.path.join(MODEL_DIR, f"hist_{model_prefix}_fold_{fold:02d}.csv"))

    save_accuracy_plot(
        df_hist, os.path.join(PLOT_DIR, f"accuracy_{model_prefix}_fold_{fold:02d}.png")
    )
    save_loss_plot(df_hist, os.path.join(PLOT_DIR, f"loss_{model_prefix}_fold_{fold:02d}.png"))


def test(df: pd.DataFrame, image_dir: str, model_prefix: str, num_channels: int, fold: int):
    """Test best performing model for the given fold and store results in a CSV file.
    Additionally, the related PR and ROC curves are stored to plots - directory.

    Args:
        df: dataframe object that contains list of images with labels and fold info
        image_dir: directory from which training images are loaded from
        model_prefix: prefix to use when storing trained models
        num_channels: number of input channels to use, 1 = NIH, 2 = ImageNet (RGB)
        fold: number indicating cross-validation fold
    """
    model = keras.models.load_model(os.path.join(MODEL_DIR, f"{model_prefix}_fold_{fold:02d}.h5"))

    _, valid_generator = get_generators(df, image_dir, num_channels, fold)
    df_test = valid_generator.get_df()

    y_test = df_test["label"].values
    y_test = y_test.astype(np.float32)
    y_pred = np.zeros((df_test.shape[0], 1))

    predict = model.predict(
        valid_generator, steps=np.ceil(df_test.shape[0] / BATCH_SIZE), verbose=1
    )
    y_pred[:, :] = predict[: df_test.shape[0]]

    save_pr_curve(y_pred, y_test, os.path.join(PLOT_DIR, f"pr_{model_prefix}_fold_{fold:02d}.png"))
    save_roc_curve(
        y_pred, y_test, os.path.join(PLOT_DIR, f"roc_{model_prefix}_fold_{fold:02d}.png")
    )

    df_predictions = df_test.copy()
    df_predictions["pred_fold" + str(fold)] = 0
    pred_col_name = "pred_fold" + str(fold)
    df_predictions.loc[:, pred_col_name] = pd.Series(np.squeeze(y_pred), index=df_predictions.index)

    predictions_csv_fn = os.path.join(MODEL_DIR, f"{model_prefix}_fold_{fold:02d}_predictions.csv")
    df_predictions.to_csv(predictions_csv_fn, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for training intermediate slice models")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--model-prefix", required=True)
    parser.add_argument("--num-channels", type=int, required=True)
    parser.add_argument("--num-folds", type=int, required=True)
    parser.add_argument(
        "--pre-train-top",
        action="store_true",
        help="Flag whether to first classifier before training whole model",
    )
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        logger.error(f"Could not find CSV file {args.input_csv}")
        sys.exit(os.EX_NOINPUT)

    df = pd.read_csv(args.input_csv)

    logger.info(f"About to train slice model using {args.num_folds} folds")
    for fold in range(args.num_folds):
        logger.info(f"Training fold {(fold+1)} / {args.num_folds}")
        train(df, args.image_dir, args.model_prefix, args.num_channels, args.pre_train_top, fold)

        logger.info(f"Testing fold {(fold+1)} / {args.num_folds}")
        test(df, args.image_dir, args.model_prefix, args.num_channels, fold)
