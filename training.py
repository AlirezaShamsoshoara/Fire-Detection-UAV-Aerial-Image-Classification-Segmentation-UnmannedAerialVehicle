"""
#################################
 Training phase after demonstration: This module uses Keras and Tensor flow to train the image classification problem
 for the labeling fire and non-fire data based on the aerial images.
 Training and Validation Data: Item 7 on https://ieee-dataport.org/open-access/flame-dataset-aerial-imagery-pile-burn-detection-using-drones-uavs
 Keras version: 2.4.0
 Tensorflow Version: 2.3.0
 GPU: Nvidia RTX 2080 Ti
 OS: Ubuntu 18.04
#################################
"""

#########################################################
# import libraries

import os.path
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from tensorflow.keras import layers


from config import new_size
from plotdata import plot_training
from config import Config_classification

#########################################################
# Global parameters and definition

data_augmentation = keras.Sequential(
        [
            layers.experimental.preprocessing.RandomFlip("horizontal"),
            layers.experimental.preprocessing.RandomRotation(0.1),
        ]
    )

image_size = (new_size.get('width'), new_size.get('height'))
batch_size = Config_classification.get('batch_size')
save_model_flag = Config_classification.get('Save_Model')
epochs = Config_classification.get('Epochs')

METRICS = [
    keras.metrics.TruePositives(name='tp'),
    keras.metrics.FalsePositives(name='fp'),
    keras.metrics.TrueNegatives(name='tn'),
    keras.metrics.FalseNegatives(name='fn'),
    keras.metrics.Accuracy(name='accuracy'),
    keras.metrics.BinaryAccuracy(name='bin_accuracy'),
    keras.metrics.Precision(name='precision'),
    keras.metrics.Recall(name='recall'),
    keras.metrics.AUC(name='auc')
]


#########################################################
# Function definition

def train_keras():
    """
    This function train a DNN model based on Keras and Tensorflow as a backend. At first, the directory of Fire and
    Non_Fire images should be defined for the model, then the model is defined, compiled and fitted over the training
    and validation set. At the end, the models is saved based on the *.h5 parameters and weights. Training accuracy and
    loss are demonstrated at the end of this function.
    :return: None, Save the trained model and plot accuracy and loss on train and validation dataset.
    """
    # This model is implemented based on the guide in Keras (Xception network)
    # https://keras.io/examples/vision/image_classification_from_scratch/
    print(" --------- Training --------- ")

    dir_fire = 'frames/Training/Fire/'
    dir_no_fire = 'frames/Training/No_Fire/'

    # 0 is Fire and 1 is NO_Fire
    fire = len([name for name in os.listdir(dir_fire) if os.path.isfile(os.path.join(dir_fire, name))])
    no_fire = len([name for name in os.listdir(dir_no_fire) if os.path.isfile(os.path.join(dir_no_fire, name))])
    total = fire + no_fire
    weight_for_fire = (1 / fire) * total / 2.0
    weight_for_no_fire = (1 / no_fire) * total / 2.0
    # class_weight = {0: weight_for_fire, 1: weight_for_no_fire}

    print("Weight for class fire : {:.2f}".format(weight_for_fire))
    print("Weight for class No_fire : {:.2f}".format(weight_for_no_fire))

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "frames/Training", validation_split=0.2, subset="training", seed=1337, image_size=image_size,
        batch_size=batch_size, shuffle=True
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "frames/Training", validation_split=0.2, subset="validation", seed=1337, image_size=image_size,
        batch_size=batch_size, shuffle=True
    )

    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(1):
        for i in range(9):
            _ = plt.subplot(3, 3, i+1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(int(labels[i]))
            plt.axis("off")

    plt.figure(figsize=(10, 10))
    for images, _ in train_ds.take(1):
        for i in range(9):
            augmented_images = data_augmentation(images)
            _ = plt.subplot(3, 3, i+1)
            plt.imshow(augmented_images[0].numpy().astype("uint8"))
            plt.axis("off")

    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)

    model = make_model_keras(input_shape=image_size + (3,), num_classes=2)
    keras.utils.plot_model(model, show_shapes=True)

    callbacks = [keras.callbacks.ModelCheckpoint("save_at_{epoch}.h5"), ]
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy", metrics=["accuracy"], )
    res_fire = model.fit(train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds, batch_size=batch_size)

    layers_len = len(model.layers)

    if save_model_flag:
        file_model_fire = 'Output/Models/model_fire_resnet_weighted_40_no_metric_simple'
        model.save(file_model_fire)
    if Config_classification.get('TrainingPlot'):
        plot_training(res_fire, 'KerasModel', layers_len)

    # Prediction on one sample frame from the test set
    img = keras.preprocessing.image.load_img(
        "frames/Training/Fire/resized_frame1801.jpg", target_size=image_size)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    predictions = model.predict(img_array)
    score = predictions[0]
    print("This image is %.2f percent Fire and %.2f percent No Fire." % (100 * (1 - score), 100 * score))


def make_model_keras(input_shape, num_classes):
    """
    This function define the DNN Model based on the Keras example.
    :param input_shape: The requested size of the image
    :param num_classes: In this classification problem, there are two classes: 1) Fire and 2) Non_Fire.
    :return: The built model is returned
    """
    inputs = keras.Input(shape=input_shape)
    # x = data_augmentation(inputs)  # 1) First option
    x = inputs  # 2) Second option

    x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(x)
    # x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.Conv2D(8, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x

    # for size in [128, 256, 512, 728]:
    for size in [8]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        residual = layers.Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)

        x = layers.add([x, residual])
        previous_block_activation = x
    x = layers.SeparableConv2D(8, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    if num_classes == 2:
        activation = "sigmoid"
        units = 1
    else:
        activation = "softmax"
        units = num_classes

    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(units, activation=activation)(x)
    return keras.Model(inputs, outputs, name="model_fire")
