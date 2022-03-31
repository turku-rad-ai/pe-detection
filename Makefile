PYTHON := python3
PIP := pip
WINDOW_SEG := 1400
LEVEL_SEG := -500
WINDOW := 700
LEVEL := 100
FOLDS = 5

STUDY_DIR := ./data_dcm
GENERATED_DIR := ./generated

ORIG_TRAIN_CSV := ./train.csv
WIP_TRAIN_CSV_PNG := ${GENERATED_DIR}/train_png.csv
TRAIN_SLICES_CSV := ${GENERATED_DIR}/train_slices.csv
TRAIN_SEQUENCES_CSV := ${GENERATED_DIR}/train_sequences.csv
RAW_PNG_DIR := ${GENERATED_DIR}/raw_pngs
AUG_PNG_DIR := ${GENERATED_DIR}/aug_pngs

# Fill in these after training with best performing folds and evaluate on some test set
EVAL_INPUT_DIR :=
EVAL_SLICE_MODEL :=
EVAL_NUM_CHANNELS :=
EVAL_SEQUENCE_MODEL :=
EVAL_OUTPUT_CSV :=


# Preprocessing
dcm_to_png:
	PYTHONHASHSEED=0 ${PYTHON} -m preprocessing.dcm_to_png \
			--study-dir=${STUDY_DIR} \
			--png-dir=${RAW_PNG_DIR} \
			--window=${WINDOW} \
			--level=${LEVEL} \
			--input-csv=${ORIG_TRAIN_CSV} \
			--output-csv=${WIP_TRAIN_CSV_PNG}

folds:
	PYTHONHASHSEED=0 ${PYTHON} -m preprocessing.assign_folds \
			--input-csv=${WIP_TRAIN_CSV_PNG} \
			--fold-count=${FOLDS} \
			--output-csv=${WIP_TRAIN_CSV_PNG}

augmentation:
	PYTHONHASHSEED=0 ${PYTHON} -m preprocessing.augmentation \
			--input-csv=${WIP_TRAIN_CSV_PNG} \
			--output-slices-csv=${TRAIN_SLICES_CSV} \
			--output-sequences-csv=${TRAIN_SEQUENCES_CSV} \
			--input-dir=${RAW_PNG_DIR} \
			--output-dir=${AUG_PNG_DIR}

######## 2D model training ####################################################
# Model A
slice_training_w_nih:
	PYTHONHASHSEED=0 ${PYTHON} -m training.train_inception_resnet_v2 \
			--input-csv=${TRAIN_SLICES_CSV} \
			--image-dir=${AUG_PNG_DIR} \
			--model-prefix=nih_slice \
			--num-channels=1 \
			--num-folds=${FOLDS}

# Model B
slice_training_w_imagenet:
	PYTHONHASHSEED=0 ${PYTHON} -m training.train_inception_resnet_v2 \
			--input-csv=${TRAIN_SLICES_CSV} \
			--image-dir=${AUG_PNG_DIR} \
			--model-prefix=imagenet_slice \
			--num-channels=3 \
			--num-folds=${FOLDS}

######## Encoding preparation #################################################
encodings_w_nih:
	PYTHONHASHSEED=0 ${PYTHON} -m training.encode_slices \
			--input-csv=${TRAIN_SEQUENCES_CSV} \
			--image-dir=${AUG_PNG_DIR} \
			--model-prefix=nih_slice \
			--num-channels=1 \
			--num-folds=${FOLDS}

encodings_w_imagenet:
	PYTHONHASHSEED=0 ${PYTHON} -m training.encode_slices \
			--input-csv=${TRAIN_SEQUENCES_CSV} \
			--image-dir=${AUG_PNG_DIR} \
			--model-prefix=imagenet_slice \
			--num-channels=3 \
			--num-folds=${FOLDS}

######## Sequence model training ##############################################
sequence_training_w_nih:
	PYTHONHASHSEED=0 ${PYTHON} -m training.train_slice_sequence_lstm \
			--input-csv=${TRAIN_SEQUENCES_CSV} \
			--model-prefix=nih_slice \
			--num-folds=${FOLDS}

sequence_training_w_imagenet:
	PYTHONHASHSEED=0 ${PYTHON} -m training.train_slice_sequence_lstm \
			--input-csv=${TRAIN_SEQUENCES_CSV} \
			--model-prefix=imagenet_slice \
			--num-folds=${FOLDS}

######## Evaluation ###########################################################
evaluation:
	PYTHONHASHSEED=0 ${PYTHON} -m evaluate \
			--input-dir=${EVAL_INPUT_DIR} \
			--window=${WINDOW} \
			--level=${LEVEL} \
			--num-channels=${EVAL_NUM_CHANNELS} \
			--slice-model=${EVAL_SLICE_MODEL} \
			--sequence-model=${EVAL_SEQUENCE_MODEL} \
			--output-csv=${EVAL_OUTPUT_CSV}


######## Test #################################################################
tests:
	${PYTHON} -m pytest ./tests

######## Clean ################################################################
clean:
	rm -rf ${GENERATED_DIR}

.PHONY: dcm_to_png \
		folds \
		augmentation \
		slice_training_w_nih \
		slice_training_w_imagenet \
		encodings_w_nih \
		encodings_w_imagenet \
		sequence_training_w_nih \
		sequence_training_w_imagenet \
		evaluation \
		tests \
		clean