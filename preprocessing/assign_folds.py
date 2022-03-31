import argparse

import pandas as pd
from sklearn.model_selection import StratifiedKFold

from pe_logger import PELogger

logger = PELogger().get_logger()

SEED = 42


def assign_folds(df: pd.DataFrame, fold_count: int) -> pd.DataFrame:
    """Assign folds to dataframe (in-place) so that each patient appears only in one fold.
    Folds are stratified based on dataset level label.

    Args:
        df: dataframe which is expected to contain following columns:
            - PatientID
            - dcm_filename
            - dataset_label

        fold_count: number of cross-validation folds to create

    Returns:
        input dataframe with pat_fold - column added
    """
    df_grp_patient = df.groupby("PatientID").first()

    # Instantiate new column "pat_fold"
    df_grp_patient["pat_fold"] = 0
    df["pat_fold"] = 0
    col_fold = df_grp_patient.columns.get_loc("pat_fold")

    # Divide folds so that they would have approx. the same amount of positive cases on dataset level
    skf = StratifiedKFold(n_splits=fold_count, random_state=SEED, shuffle=True)
    fold_counter = 0
    for _, test_index in skf.split(df_grp_patient["dcm_filename"], df_grp_patient["dataset_label"]):
        df_grp_patient.iloc[test_index, col_fold] = fold_counter
        fold_counter += 1

    df_grp_patient.reset_index(inplace=True)

    # Populate pat_fold on slice level as well
    for _, row in df_grp_patient.iterrows():
        df.loc[df["PatientID"] == row["PatientID"], "pat_fold"] = row["pat_fold"]

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True)
    parser.add_argument("--fold-count", type=int, required=True)
    parser.add_argument("--output-csv", required=True)
    args = parser.parse_args()
    df = pd.read_csv(args.input_csv)

    logger.info("About to assign folds")
    df_w_folds = assign_folds(df, args.fold_count)
    df_w_folds.to_csv(args.output_csv, index=False)
    logger.info(f"Wrote {args.output_csv}")

    logger.info("Done.")
