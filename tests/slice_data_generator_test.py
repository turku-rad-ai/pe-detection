from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
import pytest

from training.slice_data_generator import SliceDataGenerator


@pytest.fixture(scope="session")
def preprocess_fn() -> Callable[[np.ndarray], np.ndarray]:
    def passthrough(x: np.ndarray) -> np.ndarray:
        return x

    return passthrough


def test_slice_generator(tmpdir: Path, preprocess_fn: Callable[[np.ndarray], np.ndarray]):
    images_num = 5
    filename_col = "filename"
    label_col = "label"
    im_dim = (32, 32)
    batch_size = 2
    num_channels = 3
    filenames = []
    labels = []

    for i in range(images_num):
        im = np.ones((*im_dim, num_channels)) * i
        fn = tmpdir / f"image_{i:03d}.png"
        cv2.imwrite(str(fn), im)
        filenames.append(fn)
        labels.append(i % 2)

    df = pd.DataFrame(data={"filename": filenames, "label": labels})

    generator = SliceDataGenerator(
        df=df,
        image_dir=str(tmpdir),
        filename_col=filename_col,
        label_col=label_col,
        batch_size=batch_size,
        dim=im_dim,
        num_channels=num_channels,
        preprocess_fn=preprocess_fn,
        shuffle=False,
    )

    X0, y0 = generator[0]
    X1, y1 = generator[1]
    X2, y2 = generator[2]

    # Test that generator creates right amount of batches
    assert len(generator) == 3

    # Test that input tensors have right dimensions
    assert X0.shape == (batch_size, *im_dim, num_channels)
    assert X1.shape == (batch_size, *im_dim, num_channels)
    assert X2.shape == (1, *im_dim, num_channels)

    # Test that labels have right dimensions
    assert y0.shape == (batch_size, 1)
    assert y1.shape == (batch_size, 1)
    assert y2.shape == (1, 1)

    # Test that ordering is preserved when not shuffling
    assert X0[0, 0, 0, 0] == 0
    assert X0[1, 0, 0, 0] == 1
    assert X1[0, 0, 0, 0] == 2
    assert X1[1, 0, 0, 0] == 3
    assert X2[0, 0, 0, 0] == 4

    assert y0[0, 0] == 0
    assert y0[1, 0] == 1
    assert y1[0, 0] == 0
    assert y1[1, 0] == 1
    assert y2[0, 0] == 0
