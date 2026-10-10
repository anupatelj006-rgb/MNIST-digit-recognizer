import os
import tensorflow as tf
from tensorflow.keras import layers, models

os.makedirs("model", exist_ok=True)
(xtr, ytr), (xte, yte) = tf.keras.datasets.mnist.load_data()
xtr = (xtr[..., None] / 255.0).astype("float32")
xte = (xte[..., None] / 255.0).astype("float32")

aug = tf.keras.Sequential([
    layers.RandomRotation(0.12, fill_mode="constant"),   # about ±43°
    layers.RandomZoom((-0.25, 0.15), fill_mode="constant"),
    layers.RandomTranslation(0.12, 0.12, fill_mode="constant"),
])

def gauss_kernel(sigma, size=9):
    r = tf.range(-(size // 2), size // 2 + 1, dtype=tf.float32)
    g = tf.exp(-(r ** 2) / (2 * sigma ** 2))
    k = tf.tensordot(g, g, axes=0)
    return (k / tf.reduce_sum(k))[:, :, None, None]

def augment(x, y):
    x = aug(x, training=True)
    sigma = tf.random.uniform([], 0.3, 2.5)
    blurred = tf.nn.conv2d(x, gauss_kernel(sigma), 1, "SAME")
    use_blur = tf.cast(tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) < 0.6, tf.float32)
    x = use_blur * blurred + (1 - use_blur) * x
    # thicker strokes (like mouse-drawn digits) on some samples
    thick = tf.nn.max_pool2d(x, 3, 1, "SAME")
    use_thick = tf.cast(tf.random.uniform([tf.shape(x)[0], 1, 1, 1]) < 0.3, tf.float32)
    x = use_thick * thick + (1 - use_thick) * x
    x += tf.random.normal(tf.shape(x), stddev=0.06)
    x = tf.clip_by_value(x, 0, 1)
    # re-normalise brightness so faint blurred digits stay visible
    mx = tf.reduce_max(x, axis=[1, 2, 3], keepdims=True)
    return x / tf.maximum(mx, 1e-3), y

ds = (tf.data.Dataset.from_tensor_slices((xtr, ytr))
      .shuffle(20000).batch(128).map(augment, num_parallel_calls=tf.data.AUTOTUNE)
      .prefetch(tf.data.AUTOTUNE))

# validate on both clean AND augmented test data
val_ds = (tf.data.Dataset.from_tensor_slices((xte, yte)).batch(256)
          .map(augment).prefetch(tf.data.AUTOTUNE))

model = models.Sequential([
    layers.Input((28, 28, 1)),
    layers.Conv2D(32, 3, padding="same", activation="relu"), layers.BatchNormalization(),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(), layers.Dropout(0.25),
    layers.Conv2D(64, 3, padding="same", activation="relu"), layers.BatchNormalization(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(), layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(256, activation="relu"), layers.Dropout(0.4),
    layers.Dense(10, activation="softmax"),
])
model.compile(tf.keras.optimizers.Adam(1e-3),
              "sparse_categorical_crossentropy", metrics=["accuracy"])

cb = [tf.keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.5),
      tf.keras.callbacks.ModelCheckpoint("model/mnist_cnn_robust.keras",
                                         save_best_only=True, monitor="val_accuracy")]
model.fit(ds, epochs=15, validation_data=val_ds, callbacks=cb)


print("Clean test accuracy:", model.evaluate(xte, yte, verbose=0)[1])