import os.path as op
import numpy as np
from scipy import ndimage
from tqdm import tqdm

import tensorflow as tf
import tensorflow_hub as hub

EPOCHS = 5
IMAGE_SIZE = (224, 224)
MODEL_HANDLE = \
    'https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4'

out_dir = '/home/alex/Downloads/cnn'

# load pre-trained model to apply transfer learning
print('Building mobilenet_v3_small_100_224')
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=IMAGE_SIZE + (3,)),
    hub.KerasLayer(MODEL_HANDLE, trainable=True),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(2)
])
model.build((None,) + IMAGE_SIZE + (3,))
model.summary()

with np.load(op.join(out_dir, 'tmp.npz')) as Xy:
    X, y = Xy['X'], Xy['y']


# rescale images
X_zoom = np.zeros((X.shape[0],) + IMAGE_SIZE, dtype='float32')
zoom_factors = (X.shape[1] / IMAGE_SIZE[0], IMAGE_SIZE[1] / X.shape[2])
for i in range(X.shape[0])
    X_zoom[i] = ndimage.zoom(X[i, :, :, 0], zoom_factors)

np.random.seed(99)
idx = np.arange(y.size)
np.random.shuffle(idx)
split_i = int(X.shape[0] * 0.8)  # 80-20 split
train_idx, test_idx = idx[:split_i], idx[split_i:]
X_train, y_train = X[train_idx], y[train_idx]
X_test, y_test = X[test_idx], y[test_idx]

train_ds = tf.data.Dataset.from_tensor_slices(
    (X_train, y_train)).shuffle(10000).batch(32)

test_ds = tf.data.Dataset.from_tensor_slices(
    (X_test, y_test)).batch(32)

for epoch in range(EPOCHS):
    # Reset the metrics at the start of the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    for images, labels in train_ds:
        train_step(images, labels)
    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
    print(
        f'Epoch {epoch + 1}, '
        f'Loss: {train_loss.result()}, '
        f'Accuracy: {train_accuracy.result() * 100}, '
        f'Test Loss: {test_loss.result()}, '
        f'Test Accuracy: {test_accuracy.result() * 100}'
    )
