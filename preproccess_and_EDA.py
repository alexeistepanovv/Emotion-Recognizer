import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from module import *
from settings import *
#%%
data = pd.read_csv('data.csv')
data = data[data['label'] != 'Ahegao']
#%%
data['path'] = 'dataset/' + data['path']
data['size'] = data['path'].apply(lambda x: get_image_size(x))
data[['path', 'size']]
#%%
sns.countplot(x = data['label'])
#%%
class_distribution = data['label'].value_counts()
class_distribution
#%%
base_dir = BASE_DIR
input_dataset_dir = INPUT_DATASET_DIR
output_dataset_dir = OUTPUT_DATASET_DIR
#%%
for emotion in os.listdir(input_dataset_dir):
    emotion_dir = os.path.join(input_dataset_dir, emotion)
    emotion_resized_dir = os.path.join(output_dataset_dir, emotion)
    # Create a subdirectory for the emotion in `dataset_resized` if it doesn't exist
    if not os.path.exists(emotion_resized_dir):
        os.makedirs(emotion_resized_dir)

    # Loop through each image in the emotion directory
    for image_name in os.listdir(emotion_dir):
        input_image_path = os.path.join(emotion_dir, image_name)
        output_image_path = os.path.join(emotion_resized_dir, image_name)

        # Resize and save the image
        resize_and_save_image(input_image_path, output_image_path)
#%%
for emotion in os.listdir('dataset_resized'):
    emotion_dir = os.path.join('dataset_resized', emotion)
    for image_name in os.listdir(emotion_dir):
        image_path = os.path.join(emotion_dir, image_name)
        normalized_image_array = normalize_image(image_path, None, range_type='0-1')
#%%
normalized_image_array.shape
#%%
sub_dir = 'dataset_resized'
image_paths = get_image_paths(base_dir, sub_dir)
#%%
processed_images_array = process_images(image_paths)
#%%
display_sample_images(image_paths, data)