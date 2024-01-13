from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt


# Function to get the size of an image
def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size  # Returns a tuple (width, height)


def resize_and_save_image(input_path, output_path, size=(224, 224)):
    """
    Resizes an image to a specified size and saves it to a specified path.

    Parameters:
    - input_path (str): The file path to the original image.
    - output_path (str): The file path where the resized image will be saved.
    - size (tuple of int): The target size for resizing the image as (width, height).
    """
    with Image.open(input_path) as img:
        img = img.resize(size, Image.ANTIALIAS)
        img.save(output_path)


def normalize_image(image_path, range_type='0-1'):
    """
    Normalizes an image's pixel values according to the specified range.

    Parameters:
    - image_path (str): The file path to the image to be normalized.
    - range_type (str): The range to which pixel values will be normalized.
                        Either '0-1' for [0, 1] or '-1-1' for [-1, 1].

    Returns:
    - normalized_img (ndarray): The normalized image data as a NumPy array.
    """
    with Image.open(image_path) as img:
        img_array = np.array(img)
        if range_type == '0-1':
            normalized_img = img_array / 255.0
        elif range_type == '-1-1':
            normalized_img = (img_array / 127.5) - 1
        else:
            raise ValueError("range_type must be '0-1' or '-1-1'")
        return normalized_img


def get_image_paths(base_dir, sub_dir):
    """
    Retrieves a list of image file paths from a subdirectory.

    Parameters:
    - base_dir (str): The base directory path.
    - sub_dir (str): The subdirectory within the base directory.

    Returns:
    - image_paths (list of str): A list of full paths to the images.
    """
    image_paths = []
    full_path = os.path.join(base_dir, sub_dir)
    for root, dirs, files in os.walk(full_path):
        for file in files:
            if file.endswith(".png") or file.endswith(".jpg"):
                image_paths.append(os.path.join(root, file))
    return image_paths


def process_images(image_paths, size=(224, 224)):
    processed_images = []
    for image_path in image_paths:
        with Image.open(image_path) as img:
            # Resize the image
            img = img.resize(size, Image.ANTIALIAS)
            # Normalize the image
            img_array = np.array(img) / 255.0
            # Append the processed image to the list
            processed_images.append(img_array)
    return np.array(processed_images)


def display_sample_images(data, num_samples=3):
    """
    Displays sample images from each class.

    Parameters:
    - data: DataFrame containing image paths and labels
    - num_samples: number of samples to display per class
    """
    unique_classes = data['label'].unique()
    fig, axes = plt.subplots(nrows=len(unique_classes), ncols=num_samples,
                             figsize=(5 * num_samples, 5 * len(unique_classes)))

    for i, emotion in enumerate(unique_classes):
        sample_images = data[data['label'] == emotion].sample(num_samples)
        for j in range(num_samples):
            img_path = sample_images.iloc[j]['path']
            image = Image.open(img_path)
            ax = axes[i, j]
            ax.imshow(image)
            ax.axis('off')
            ax.set_title(emotion)

    plt.tight_layout()
    plt.show()
