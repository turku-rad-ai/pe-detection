from typing import List

import pandas as pd
import pytest

from preprocessing.assign_folds import assign_folds

testdata = [
    [
        [
            "patient1",
            "patient2",
            "patient3",
            "patient4",
            "patient5",
            "patient6",
            "patient7",
            "patient8",
            "patient9",
            "patient1",  # second 1
            "patient3",  # second 3
            "patient10",
        ],
        [
            "image1.dcm",
            "image2.dcm",
            "image3.dcm",
            "image4.dcm",
            "image5.dcm",
            "image6.dcm",
            "image7.dcm",
            "image8.dcm",
            "image9.dcm",
            "image10.dcm",
            "image11.dcm",
            "image12.dcm",
        ],
        [1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1],
        3,
    ]
]


@pytest.mark.parametrize("patient_ids,dcm_filenames,dataset_labels,folds", testdata)
def test_assign_folds(
    patient_ids: List[str],
    dcm_filenames: List[str],
    dataset_labels: List[int],
    folds: int,
):
    data = {
        "PatientID": patient_ids,
        "dcm_filename": dcm_filenames,
        "dataset_label": dataset_labels,
    }

    df = pd.DataFrame(data=data)

    df = assign_folds(df, fold_count=folds)

    # pat_fold - column must have been added
    assert "pat_fold" in df.columns

    # Check that folds are on proper range
    assert df["pat_fold"].min() == 0
    assert df["pat_fold"].max() == folds - 1

    # Test that each patient belongs to one and only one fold
    assert min([item.shape[0] for item in list(df.groupby("PatientID")["pat_fold"].unique())]) == 1
    assert max([item.shape[0] for item in list(df.groupby("PatientID")["pat_fold"].unique())]) == 1
