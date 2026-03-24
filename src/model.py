import tensorflow as tf
from tensorflow.keras import callbacks, layers, models


def build_model(num_classes: int, img_size=(224, 224), dropout_rate: float = 0.3):
    data_augmentation = tf.keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.1),
        ]
    )

    base_model = tf.keras.applications.MobileNetV3Large(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    model = models.Sequential(
        [
            layers.Input(shape=(img_size[0], img_size[1], 3)),
            data_augmentation,
            layers.Lambda(tf.keras.applications.mobilenet_v3.preprocess_input),
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(dropout_rate),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_training_callbacks():
    lr_scheduler = callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1,
    )
    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        restore_best_weights=True,
        verbose=1,
    )
    return [lr_scheduler, early_stopping]
