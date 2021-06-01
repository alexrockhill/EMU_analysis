import os
import os.path as op
import numpy as np
from subprocess import call

import mne
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D

import matplotlib.pyplot as plt


data_dir = '/home/alex/Downloads/spectrograms'
out_dir = '/home/alex/Downloads/cnn'
data = [op.join(data_dir, f) for f in os.listdir(data_dir)
        if op.splitext(f)[-1] == '.npz']
events = [('Fixation', 'Response'), ('Fixation', 'Go Cue')]

ch_data = '/home/alex/Downloads/spectrograms/sub-1_ch-LPM8_spectrogram.npz'

for ch_data in data:
    # load data, make a copy for fast reading
    with np.load(ch_data) as data_dict:
        data_dict = {k: v for k, v in data_dict.items()}
    # filter DC signal component before decimating
    for bl_event, event in events:
        info = mne.create_info(['DC'], data_dict['sfreq'], 'seeg')
        for e in (bl_event, event):
            epochs = mne.EpochsArray(data_dict[e][:, 0:1], info)
            epochs.filter(l_freq=None, h_freq=80)
            data_dict[e][:, 0:1] = epochs._data
        info = mne.create_info(list(data_dict['freqs'].astype(str)),
                               data_dict['sfreq'], 'seeg')
        bl_epochs = mne.EpochsArray(data_dict[bl_event], info).decimate(20)
        epochs = mne.EpochsArray(data_dict[event], info).decimate(20)
        X = np.concatenate([bl_epochs._data, epochs._data], axis=0)
        X = X[:, :, :, np.newaxis]  # add axis for color channel (unused)
        X = X.astype('float32')
        y = np.concatenate([np.repeat(0, data_dict[bl_event].shape[0]),
                            np.repeat(1, data_dict[bl_event].shape[0])])
        y = y[:, np.newaxis]  # add axis for y as well
        np.savez_compressed(op.join(out_dir, 'tmp.npz'), X=X, y=y)
        call()
        os.remove('tmp.npz')


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

model = tf.keras.models.Sequential()
model.add(Conv2D(32, 3, activation='relu',
                 input_shape=X.shape[1:]))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2))

'''
tf.keras.layers.MaxPooling2D(2,),
tf.keras.layers.Conv2D(64, 3, activation='relu'),
tf.keras.layers.MaxPooling2D(2),
tf.keras.layers.Conv2D(64, 3, activation='relu'),
'''


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_ds, epochs=10,
                    validation_data=test_ds)
fig, ax = plt.subplots()
ax.plot(history.history['accuracy'], label='accuracy')
ax.plot(history.history['val_accuracy'], label='val_accuracy')
ax.set_xlabel('Epoch')
ax.set_ylabel('Accuracy')
ax.set_ylim([0, 1])
ax.legend(loc='lower right')
fig.show()

loss, acc = model.evaluate(X_test, y_test)

model = tf.keras.applications.VGG16(weights='imagenet')
_img = tf.keras.preprocessing.image.load_img(
    '/home/alex/Downloads/1.jpeg', target_size=(224, 224))
plt.imshow(_img)
plt.show()
img = tf.keras.preprocessing.image.img_to_array(_img)
img = img.reshape((1, *img.shape))
y_pred = model.predict(img)

images = tf.Variable(img, dtype=float)

with tf.GradientTape() as tape:
    pred = model(images, training=False)
    class_idxs_sorted = np.argsort(pred.numpy().flatten())[::-1]
    loss = pred[0][class_idxs_sorted[0]]

grads = tape.gradient(loss, images)
dgrad_abs = tf.math.abs(grads)
dgrad_max_ = np.max(dgrad_abs, axis=3)[0]
arr_min, arr_max = np.min(dgrad_max_), np.max(dgrad_max_)
grad_eval = (dgrad_max_ - arr_min) / (arr_max - arr_min + 1e-18)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].imshow(_img)
i = axes[1].imshow(grad_eval, cmap='jet', alpha=0.8)
fig.colorbar(i)
