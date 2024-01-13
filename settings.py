import os

BASE_DIR = 'C:/Desktop/emotion_recognizer/'
INPUT_DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
OUTPUT_DATASET_DIR = os.path.join(BASE_DIR, 'dataset_resized')

DATASET_DIR = OUTPUT_DATASET_DIR
BATCH_SIZE = 64
IMAGE_SIZE = (224, 224)
VALIDATION_SPLIT_RATIO = 0.2
SEED = 42