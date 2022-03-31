# Automated Detection of Pulmonary Embolism from CT-Angiograms using Deep Learning

## Abstract

BACKGROUND: The aim of this study was to develop and evaluate a deep neural network model in the automated detection of pulmonary embolism (PE) from computed tomography pulmonary angiograms (CTPAs) using only weakly labelled training data.

METHODS: We developed a deep neural network model consisting of two parts: a convolutional neural network architecture called InceptionResNet V2 and a long-short term memory network to process whole CTPA stacks as sequences of slices. Two versions of the model were created using either chest X rays (Model A) or natural images (Model B) as pre-training data. We retrospectively collected 600 CTPAs to use in training and validation and 200 CTPAs to use in testing. CTPAs were annotated only with binary labels on both stack- and slice-based levels. Performance of the models was evaluated with ROC and precision-recall curves, specificity, sensitivity, accuracy, as well as positive and negative predictive values.

RESULTS: Both models performed well on both stack- and slice-based levels. On the stack-based level, Model A reached specificity and sensitivity of 93.5% and 86.6%, respectively, outperforming Model B slightly (specificity 90.7% and sensitivity 83.5%). However, the difference between their ROC AUC scores was not statistically significant (0.94 vs 0.91, p = 0.07).

CONCLUSIONS: We show that a deep learning model trained with a relatively small, weakly annotated dataset can achieve excellent performance results in detecting PE from CTPAs.

Link to the paper: \
[Automated Detection of Pulmonary Embolism from CT-Angiograms using Deep Learning](https://bmcmedimaging.biomedcentral.com/track/pdf/10.1186/s12880-022-00763-z.pdf)

```
@article{Huhtanen:2022,
  author = {Huhtanen, Heidi and Nyman, Mikko and Mohsen, Tarek and Virkki, Arho and Karlsson, Antti and Hirvonen, Jussi},
  year = {2022},
  title = {Automated detection of pulmonary embolism from CT-angiograms using deep learning},
  volume = {22},
  journal = {BMC Medical Imaging},
  doi = {10.1186/s12880-022-00763-z}
}
```

## Requirements

Install requirements:

```
python3 -m venv venv
source venv/bin/activate
pip3 install -r requirements.txt
```

Tested on Ubuntu 20.04, python 3.8, TensorFlow 2.8.0

## Data

This repository expects data to be available in DICOM CT Image Storage format in `data_dcm` - directory.
The models described in the paper have been trained using 3 mm axial slices. The LSTM model considers
only the first 96 slices for detection and disregards the rest.

DICOM images used as input should be located in a directory with the following structure:

```
<data_dcm - dir>
  <StudyInstanceUID>
    <SeriesInstanceUID>
      <DICOM - file>
```

The DICOM files should be accompanied with related metadata and annotations. Specifically, a `train.csv` - file with following columns
should be placed in the root-directory:

- PatientID
- StudyInstanceUID
- SeriesInstanceUID
- SOPInstanceUID
- dcm_filename
- label
- dataset_label

### Inception-ResNet V2 weights

The code makes use of Inception-ResNet V2 architecture for classifying 2D slices. Two sets of weights are being used as basis for training:

- NIH chest X-rays (Model A)
  - These weights have been trained as part of the 1st place solution for RSNA Pneumonia Detection Challenge. [Weights](https://docs.google.com/uc?export=download&id=1rI_WSlot6ZNa_ERdLSCsGquUXEK_ikYb) for the model can be downloaded from the link published in the original [repository](https://github.com/i-pan/kaggle-rsna18/). Place the _InceptionResNetV2_NIH15_Px256.h5_ file under `pretrained` - directory
- ImageNet based weights (Model B)
  - These weights are directly loaded from Keras and do not need to be downloaded manually

### Preparations

```
make dcm_to_png
make folds
make augmentation
```

## Training

In the first phase intermediate 2D models are trained to do binary classification for presence of PE on slice level. These models are then used for creating encodings which serve as inputs for training another, LSTM based, sequence model. The sequence model predicts presence of PE for the whole dataset and additionally predicts labels for individual slices for better interpretability.

All models are trained using 5 fold cross-validation from which the best performing folds are used for evaluation.

### 2D Slice model training

```bash
make slice_training_w_nih
make slice_training_w_imagenet
```

### Prepare slice encodings

```bash
make encodings_w_nih
make encodings_w_imagenet
```

### LSTM sequence model training

```bash
make sequence_training_w_nih
make sequence_training_w_imagenet
```

## Evaluation

Fill-in data directory and models to the Makefile and then run the following:

```bash
make evaluation
```

## Acknowledgements

This repository makes use of Inception-ResNet V2 weights trained on NIH chest X-ray dataset from [kaggle-rsna18](https://github.com/i-pan/kaggle-rsna18/) repository.
