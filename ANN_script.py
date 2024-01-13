from module import get_image_size
from settings import *
from model_functions import *
#%%
data = pd.read_csv('data.csv')
data = data[data['label'] != 'Ahegao']
data['path'] = 'dataset_resized/' + data['path']
data['size'] = data['path'].apply(lambda x: get_image_size(x))
data[['path', 'size']]
#%%
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VALIDATION_SPLIT_RATIO,
    subset="training",
    seed=100,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)
validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_DIR,
    validation_split=VALIDATION_SPLIT_RATIO,
    subset="validation",
    seed=200,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
)
#%%
train_size = len(train_dataset)
val_size = len(validation_dataset)
total = train_size + val_size
#%%
class_names=train_dataset.class_names
#%%
vgg16_model = build_vgg16_model(num_classes=5)
vgg16_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_vgg16 = train_model(vgg16_model, train_dataset, validation_dataset, epochs=40)
plot_diagnostics(history_vgg16, model_name='vgg16')
save_training_results(history_vgg16, 'vgg16')
#%%
resnet_model = build_resnet_model(num_classes=5)
resnet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_resnet = train_model(resnet_model, train_dataset, validation_dataset, epochs=40)
plot_diagnostics(history_resnet, model_name='resnet')
save_training_results(history_resnet, 'resnet')
#%%
effnetv2_model = build_effnetv2_model(num_classes=5)
effnetv2_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history_effnetv2 = train_model(effnetv2_model, train_dataset, validation_dataset, epochs=40)
plot_diagnostics(history_effnetv2, model_name='effnetv2')
save_training_results(history_effnetv2, 'effnetv2')

