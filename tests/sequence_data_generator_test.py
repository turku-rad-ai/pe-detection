import numpy as np
import pandas as pd

from training.sequence_data_generator import SequenceDataGenerator


def test_sequence_generator():
    filename_col = "png_filename"
    label_col = "label"
    dataset_label_col = "dataset_label"
    batch_size = 2
    encoding_dim = 3
    num_slices = 2

    png_filenames = [
        "file1.png",
        "file2.png",
        "file3.png",
        "file4.png",
        "file5.png",
        "file6.png",
        "file7.png",
    ]

    series_instance_uids = [
        "series1",
        "series1",
        "series2",
        "series2",
        "series2",
        "series3",
        "series3",
    ]

    instance_numbers = [2, 1, 1, 3, 2, 3, 4]
    slice_labels = [0, 0, 1, 0, 1, 1, 0]
    dataset_labels = [0, 0, 1, 1, 1, 1, 1]

    encodings = np.zeros((len(png_filenames), encoding_dim), dtype=np.float32)
    for i in range(len(png_filenames)):
        encodings[i, :] = i

    encoding_dict = dict(zip(png_filenames, encodings))

    data = {
        "png_filename": png_filenames,
        "SeriesInstanceUID": series_instance_uids,
        "InstanceNumber": instance_numbers,
        "label": slice_labels,
        "dataset_label": dataset_labels,
    }

    df = pd.DataFrame(data=data)

    generator = SequenceDataGenerator(
        df,
        encoding_dict,
        filename_col,
        label_col,
        dataset_label_col,
        batch_size,
        encoding_dim,
        num_slices,
        shuffle=False,
    )

    X0, y0 = generator[0]
    X1, y1 = generator[1]

    # Test that generator creates right amount of batches
    assert len(generator) == 2

    # Test that input tensors have right dimensions
    assert X0.shape == (batch_size, num_slices, encoding_dim)
    assert X1.shape == (1, num_slices, encoding_dim)  # Last one is not full

    # Test that sorting based on InstanceNumber works
    assert X0[0, 0, 0] == 1
    assert X0[0, 1, 0] == 0

    assert X0[1, 0, 0] == 2
    assert X0[1, 1, 0] == 4

    assert X1[0, 0, 0] == 5
    assert X1[0, 1, 0] == 6

    # Test that labels are given for both slices and stacks
    assert len(y0) == 2
    assert len(y1) == 2

    # Test that labels have right dimensions
    assert y0[0].shape == (batch_size, num_slices, 1)  # slice label
    assert y0[1].shape == (batch_size, 1)  # dataset label

    assert y1[0].shape == (1, num_slices, 1)  # slice label
    assert y1[1].shape == (1, 1)  # dataset label
