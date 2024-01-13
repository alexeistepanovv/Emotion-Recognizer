import pandas as pd

import datetime
import os

import matplotlib.pyplot as plt

import tensorflow as tf
from keras import layers, models
from keras.callbacks import EarlyStopping, ModelCheckpoint

from settings import IMAGE_SIZE


def train_model(model, train_ds, val_ds, epochs=40):
    early_stopping = EarlyStopping(patience=5, restore_best_weights=True)
    checkpoint = ModelCheckpoint(f"{model.name}_best.h5", save_best_only=True)
    history = model.fit(
        train_ds,
        epochs=epochs,
        validation_data=val_ds,
        callbacks=[checkpoint, early_stopping]
    )
    return history


def plot_diagnostics(history, model_name):
    # Create 'plots' directory if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    # Current date and time for filename
    current_time = datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")
    # Plotting
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Save plot to the 'plots' directory
    plot_filename = f"plots/{model_name}_{current_time}.png"
    plt.savefig(plot_filename)
    plt.show()


def save_training_results(history, model_name):
    # Create a DataFrame from the history object
    df = pd.DataFrame(history.history)
    df['epoch'] = range(1, len(df) + 1)

    # Save the DataFrame to a CSV file
    results_file = f'results_{model_name}.csv'
    df.to_csv(results_file, index=False)
    print(f"Training results saved to {results_file}")


def build_vgg16_model(num_classes):
    base_model = tf.keras.applications.VGG16(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the convolutional base

    vgg16_model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dense(4096, activation='relu'),
        layers.Dense(4096, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])

    return vgg16_model


def build_resnet_model(num_classes):
    base_model = tf.keras.applications.ResNet50(input_shape=(*IMAGE_SIZE, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the convolutional base

    resnet_model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(1024, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    return resnet_model


def build_effnetv2_model(num_classes):
    base_model = tf.keras.applications.EfficientNetV2M(input_shape=(*IMAGE_SIZE, 3),
                                                       include_top=False, weights='imagenet')
    base_model.trainable = False
    model = models.Sequential([
        base_model,
        layers.Flatten(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model
