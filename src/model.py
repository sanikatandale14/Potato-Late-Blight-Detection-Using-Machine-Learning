import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Dense, Flatten, Dropout,
    BatchNormalization, GlobalAveragePooling2D, Input
)
from tensorflow.keras.regularizers import l2
from config.settings import IMAGE_SIZE, NUM_CLASSES, LEARNING_RATE


def create_custom_cnn(input_shape=None, num_classes=None, learning_rate=None):
    if input_shape is None:
        input_shape = (*IMAGE_SIZE, 3)
    if num_classes is None:
        num_classes = NUM_CLASSES
    if learning_rate is None:
        learning_rate = LEARNING_RATE

    model = Sequential([
        Input(shape=input_shape),

        Conv2D(32, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Conv2D(32, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.15),

        Conv2D(64, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.15),

        Conv2D(128, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),

        Conv2D(256, (3, 3), activation='relu', padding='same',
               kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        MaxPooling2D((2, 2)),
        Dropout(0.2),

        Flatten(),
        Dense(512, activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(256, activation='relu', kernel_regularizer=l2(0.0001)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def create_transfer_learning_model(base_model_name='MobileNetV2', input_shape=None,
                                   num_classes=None, learning_rate=None, trainable=False):
    if input_shape is None:
        input_shape = (*IMAGE_SIZE, 3)
    if num_classes is None:
        num_classes = NUM_CLASSES
    if learning_rate is None:
        learning_rate = LEARNING_RATE

    base_models = {
        'MobileNetV2': tf.keras.applications.MobileNetV2,
        'ResNet50': tf.keras.applications.ResNet50,
        'EfficientNetB0': tf.keras.applications.EfficientNetB0,
    }

    if base_model_name not in base_models:
        base_model_name = 'MobileNetV2'

    base_model = base_models[base_model_name](
        weights='imagenet',
        include_top=False,
        input_shape=input_shape
    )

    base_model.trainable = trainable

    inputs = Input(shape=input_shape)
    x = base_model(inputs, training=trainable)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.0001))(x)
    x = Dropout(0.2)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
