import argparse
import os
import random as rn
import sys

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import config
from pe_logger import PELogger
from plots.plot_results import save_accuracy_plot, save_loss_plot, save_pr_curve, save_roc_curve
from training.sequence_data_generator import SequenceDataGenerator

# Set seeding based on Keras documentation
np.random.seed(1)
rn.seed(2)
tf.random.set_seed(3)

logger = PELogger().get_logger()

CONFIG = config.config()
MODEL_DIR = CONFIG["model"]["model_dir"]
PLOT_DIR = CONFIG["model"]["plot_dir"]
ENCODING_CSV = CONFIG["model"]["encoding_csv"]
ENCODING_DIR = CONFIG["model"]["encoding_dir"]
ENCODING_DIM = CONFIG["model"]["encoding_dim"]
FILENAME_COL = "png_filename"
LABEL_COL = "label"
DATASET_LABEL_COL = "dataset_label"
FOLD_COL = "pat_fold"
NUM_SLICES = CONFIG["model"]["sequence_num_slices"]
BATCH_SIZE = 16
EPOCHS = 24


def get_data(df_orig: pd.DataFrame, num_slices: int, fold: int):
    """Get dataframes for training and validation based on given fold"""

    # Filter slices that do not fit to our sequence model (order > num_slices)
    df = df_orig.sort_values(by=["SeriesInstanceUID", "InstanceNumber"])
    df_grp = df.groupby("SeriesInstanceUID")
    df = pd.concat([g[1][:num_slices] for g in list(df_grp)], axis=0, ignore_index=True)

    # Remove series that have invalid slices (black frames with positive label)
    invalid_series = df.loc[~df["valid"], "SeriesInstanceUID"].values
    df_valid_no_aug = df.loc[~df["SeriesInstanceUID"].isin(invalid_series) & (df["augmented"] == 0)]

    df_test = df_valid_no_aug.loc[df["pat_fold"] == fold]
    df_train = df_valid_no_aug.loc[df["pat_fold"] != fold]

    return df_train, df_test


def get_model() -> keras.models.Model:
    """Get a custom LSTM based model for classifying a set of slice encodings.
    The model predicts presence of PE at both slice and stack level.
    """
    tf.keras.backend.clear_session()

    slice_stack = keras.Input(shape=(NUM_SLICES, ENCODING_DIM))

    x = layers.BatchNormalization()(slice_stack)
    x = layers.Bidirectional(layers.LSTM(units=64, return_sequences=True))(x)
    x = layers.TimeDistributed(layers.Dense(512, activation="relu"))(x)
    x = layers.TimeDistributed(layers.Dense(128, activation="relu"))(x)
    slice_outputs = layers.TimeDistributed(layers.Dense(1, activation="sigmoid"), name="slices")(x)

    reshaped_features = layers.Reshape((NUM_SLICES,))(slice_outputs)
    reshaped_features = layers.Dropout(0.05)(reshaped_features)
    stack_output = layers.Dense(1, activation="sigmoid", name="sequence")(reshaped_features)

    model = keras.models.Model(inputs=slice_stack, outputs=[slice_outputs, stack_output])

    return model


def train_model(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    model_prefix: str,
    enc_filenames: np.ndarray,
    fold: int,
):
    """Train a sequence model for classifying a set of slice encodings for PE.
    Best model for given fold will be saved based validation loss (slice + stack loss).

    Args:
        df_train: dataframe object that contains list of images
                  with labels and fold info for training
        df_test: dataframe object that contains list of images
                 with labels and fold info for validation
        model_prefix: prefix to use when storing trained models
        enc_filenames: array of filenames used as keys to retrive encodings
        fold: number indicating cross-validation fold
    """
    encodings_npy_fn = os.path.join(ENCODING_DIR, model_prefix + f"_encodings_fold_{fold:02d}.npy")
    model_fn = os.path.join(MODEL_DIR, f"{model_prefix}_lstm_fold_{fold:02d}.h5")

    model = get_model()
    if fold == 0:
        model.summary()

    # Create a dictionary for generators that maps png filenames to encodings
    encodings = np.load(encodings_npy_fn)
    encoding_dict = dict(zip(enc_filenames, encodings))

    opt = keras.optimizers.Nadam(learning_rate=0.001)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    mc = keras.callbacks.ModelCheckpoint(
        model_fn,
        monitor="val_loss",
        verbose=1,
    )

    train_generator = SequenceDataGenerator(
        df_train,
        encoding_dict,
        FILENAME_COL,
        LABEL_COL,
        DATASET_LABEL_COL,
        BATCH_SIZE,
        ENCODING_DIM,
        NUM_SLICES,
        shuffle=True,
    )

    valid_generator = SequenceDataGenerator(
        df_test,
        encoding_dict,
        FILENAME_COL,
        LABEL_COL,
        DATASET_LABEL_COL,
        BATCH_SIZE,
        ENCODING_DIM,
        NUM_SLICES,
        shuffle=False,
    )

    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=EPOCHS,
        validation_data=valid_generator,
        validation_steps=len(valid_generator),
        callbacks=[mc],
    )

    df_hist = pd.DataFrame(history.history)
    df_hist.to_csv(os.path.join(MODEL_DIR, f"hist_{model_prefix}_seq_fold_{fold:02d}.csv"))
    _save_train_history_plots(df_hist, model_prefix, fold)

    # Save prediction scores for further analysis and plotting
    model = keras.models.load_model(model_fn)

    df_test_series = df_test.groupby("SeriesInstanceUID").first()

    generator = SequenceDataGenerator(
        df_test,
        encoding_dict,
        FILENAME_COL,
        LABEL_COL,
        DATASET_LABEL_COL,
        BATCH_SIZE,
        ENCODING_DIM,
        NUM_SLICES,
        shuffle=False,
    )

    predict = model.predict(
        generator, steps=np.ceil(df_test_series.shape[0] / BATCH_SIZE), verbose=1
    )

    slice_preds = predict[0].flatten()
    stack_preds = predict[1].flatten()

    # Save dataframe for slices
    df_predictions = df_test.copy()
    slice_preds = slice_preds[: df_test.shape[0]]
    df_predictions["pred_fold" + str(fold)] = slice_preds
    slice_csv_filename = os.path.join(
        MODEL_DIR, f"{model_prefix}_lstm_fold_{fold:02d}_predictions.csv"
    )
    df_predictions.to_csv(slice_csv_filename, index=False)
    save_pr_curve(
        slice_preds,
        df_test["label"],
        os.path.join(PLOT_DIR, f"pr_{model_prefix}_lstm_fold_{fold:02d}.png"),
    )
    save_roc_curve(
        slice_preds,
        df_test["label"],
        os.path.join(PLOT_DIR, f"roc_{model_prefix}_lstm_fold_{fold:02d}.png"),
    )

    # Save dataframe for datasets
    df_predictions_stack = df_test_series.copy()
    stack_preds = stack_preds[: df_test_series.shape[0]]
    df_predictions_stack["pred_fold_stack" + str(fold)] = stack_preds
    stack_csv_filename = os.path.join(
        MODEL_DIR, f"{model_prefix}_stack_lstm_fold_{fold:02d}_predictions.csv"
    )
    df_predictions_stack.to_csv(stack_csv_filename, index=False)
    stack_true = df_test.groupby("SeriesInstanceUID").first()["dataset_label"]

    save_pr_curve(
        stack_preds,
        stack_true,
        os.path.join(PLOT_DIR, f"pr_{model_prefix}_stack_lstm_fold_{fold:02d}.png"),
    )
    save_roc_curve(
        stack_preds,
        stack_true,
        os.path.join(PLOT_DIR, f"roc_{model_prefix}_stack_lstm_fold_{fold:02d}.png"),
    )


def _save_train_history_plots(df_hist: pd.DataFrame, model_prefix: str, fold: int):
    """Save training history as simple plots"""

    # Slice accuracy
    save_accuracy_plot(
        df_hist,
        os.path.join(
            PLOT_DIR,
            f"slice_accuracy_{model_prefix}_seq_fold_{fold:02d}.png",
        ),
        "slices_accuracy",
        "val_slices_accuracy",
    )
    # Slice loss
    save_loss_plot(
        df_hist,
        os.path.join(
            PLOT_DIR,
            f"slice_loss_{model_prefix}_seq_fold_{fold:02d}.png",
        ),
        "slices_loss",
        "val_slices_loss",
    )
    # Stack accuracy
    save_accuracy_plot(
        df_hist,
        os.path.join(
            PLOT_DIR,
            f"stack_accuracy_{model_prefix}_seq_fold_{fold:02d}.png",
        ),
        "sequence_accuracy",
        "val_sequence_accuracy",
    )
    # Stack loss
    save_loss_plot(
        df_hist,
        os.path.join(
            PLOT_DIR,
            f"stack_loss_{model_prefix}_seq_fold_{fold:02d}.png",
        ),
        "sequence_loss",
        "val_sequence_loss",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for training PE sequence model")
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--model-prefix", required=True)
    parser.add_argument("--num-folds", type=int, required=True)
    args = parser.parse_args()

    if not os.path.exists(args.input_csv):
        logger.error(f"Could not find training CSV file {args.input_csv}")
        sys.exit(os.EX_NOINPUT)

    enc_filenames_csv = os.path.join(ENCODING_DIR, ENCODING_CSV)
    if not os.path.exists(enc_filenames_csv):
        logger.error(f"Could not find encoding filename CSV file {enc_filenames_csv}")
        sys.exit(os.EX_NOINPUT)

    df = pd.read_csv(args.input_csv)
    df_enc_filenames = pd.read_csv(enc_filenames_csv)

    logger.info(f"About to train sequence model using {args.num_folds} folds")
    for fold in range(args.num_folds):
        logger.info(f"Training fold {(fold+1)} / {args.num_folds}")

        df_train, df_test = get_data(df, NUM_SLICES, fold)
        train_model(
            df_train, df_test, args.model_prefix, df_enc_filenames["png_filename"].values, fold
        )
