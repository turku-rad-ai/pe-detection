import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from funcsigs import signature


logging.getLogger("matplotlib.font_manager").disabled = True


def save_accuracy_plot(
    df_hist: pd.DataFrame,
    filename: str,
    acc_col: str = "accuracy",
    val_acc_col: str = "val_accuracy",
):
    x_values = range(1, df_hist.shape[0] + 1)

    plt.figure()
    plt.plot(x_values, df_hist[acc_col])
    plt.plot(x_values, df_hist[val_acc_col])
    plt.title("Model Accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.xticks(x_values)
    plt.legend(["train", "valid"], loc="upper left")
    plt.savefig(filename)  # can be .png, .pdf, .svg
    plt.close()


def save_loss_plot(
    df_hist: pd.DataFrame, filename: str, loss_col: str = "loss", val_loss_col: str = "val_loss"
):
    x_values = range(1, df_hist.shape[0] + 1)

    plt.figure()
    plt.plot(x_values, df_hist[loss_col])
    plt.plot(x_values, df_hist[val_loss_col])
    plt.title("Model Loss")
    plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.xticks(x_values)
    plt.legend(["train", "valid"], loc="upper left")
    plt.savefig(filename)  # can be .png, .pdf, .svg
    plt.close()


def save_pr_curve(y_pred: np.ndarray, y_test: np.ndarray, filename: str):
    precision, recall, _ = precision_recall_curve(y_test, y_pred)

    step_kwargs = {"step": "post"} if "step" in signature(plt.fill_between).parameters else {}

    plt.figure()
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.step(recall, precision, color="b", alpha=0.2, where="post")
    plt.fill_between(recall, precision, alpha=0.2, color="b", **step_kwargs)

    plt.title("Precision / Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.savefig(filename)  # can be .png, .pdf, .svg
    plt.close()


def save_roc_curve(y_pred: np.ndarray, y_test: np.ndarray, filename: str):
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(
        fpr, tpr, color="darkorange", lw=lw, label="ROC curve (area = {0:.2f})".format(roc_auc)
    )
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver operating characteristic")
    plt.legend(loc="lower right")
    plt.savefig(filename)  # can be .png, .pdf, .svg
    plt.close()
