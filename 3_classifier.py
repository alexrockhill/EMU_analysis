import os
import os.path as op
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA

import mne
import matplotlib.pyplot as plt

n_components = 50
bl_event, event = 'Fixation', 'Response'
data_dir = './derivatives/spectrograms'
out_dir = f'./derivatives/pca_{event.lower()}_classifier'

if not op.isdir(out_dir):
    os.makedirs(out_dir)


def plot_image(fig, ax, img, data_dict, vmin=None, vmax=None,
               cmap='RdYlBu_r', cbar=True):
    vmin = img.min() if vmin is None else vmin
    vmax = img.max() if vmax is None else vmax
    c = ax.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
    ax.invert_yaxis()
    ax.set_xlabel('Time (s)')
    ticks = np.linspace(0, data_dict[event + ' Times'].size - 1,
                        5).round().astype(int)
    ax.set_xticks(np.linspace(0, epochs.times.size, 5))
    ax.set_xticklabels(data_dict[event + ' Times'][ticks].round(2))
    ax.set_ylabel('Frequency (Hz)')
    ax.set_yticks(range(len(data_dict['freqs'])))
    ax.set_yticklabels([f'{f}        ' if i % 2 else f for i, f in
                        enumerate(data_dict['freqs'].round(
                        ).astype(int))], fontsize=8)
    if cbar:
        fig.colorbar(c)


data = [op.join(data_dir, f) for f in os.listdir(data_dir)
        if op.splitext(f)[-1] == '.npz']

with np.load(data[0]) as data_dict:
    data_dict = {k: v for k, v in data_dict.items()}

_, n_freqs, n_times = data_dict[event].shape

scores = dict()
imgs = dict()
feature_map = None
for ch_data in tqdm(data):
    sub, ch = [phrase.split('-')[1] for phrase in
               op.basename(ch_data).split('_')[0:2]]
    out_fname = f'sub-{sub}_ch-{ch}_{bl_event}-{event}'
    out_fname = out_fname.replace(' ', '_')
    # load data, make a copy for fast reading
    with np.load(ch_data) as data_dict:
        data_dict = {k: v for k, v in data_dict.items()}
    # filter DC signal component before decimating
    info = mne.create_info(['DC'], data_dict['sfreq'], 'seeg')
    for e in (bl_event, event):
        epochs = mne.EpochsArray(data_dict[e][:, 0:1], info)
        epochs.filter(l_freq=None, h_freq=80)
        data_dict[e][:, 0:1] = epochs._data
    info = mne.create_info(list(data_dict['freqs'].astype(str)),
                           data_dict['sfreq'], 'seeg')
    bl_epochs = mne.EpochsArray(
        data_dict[bl_event], info, baseline=None)
    epochs = mne.EpochsArray(
        data_dict[event], info, baseline=None)
    X = np.concatenate([bl_epochs._data, epochs._data], axis=0)
    X = X.reshape(X.shape[0], -1).astype('float32')  # flatten features
    y = np.concatenate([np.repeat(0, data_dict[bl_event].shape[0]),
                        np.repeat(1, data_dict[event].shape[0])])
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.2, random_state=99)
    pca = PCA(n_components=n_components, svd_solver='randomized',
              whiten=True).fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    classifier = SVC(kernel='linear')
    np.random.seed(99)
    classifier.fit(X_train_pca, y_train)
    score = classifier.score(X_test_pca, y_test)
    scores[f'sub-{sub}_ch-{ch}'] = score
    scores[f'sub-{sub}_ch-{ch}_n_epochs'] == y_test.size
    eigenvectors = pca.components_.reshape(
        (n_components, n_freqs, n_times))
    img = np.sum(classifier.coef_[0][:, np.newaxis, np.newaxis] * eigenvectors,
                 axis=0)
    imgs[f'sub-{sub}_ch-{ch}'] = img
    if score > 0.75:
        fig, ax = plt.subplots()
        fig.suptitle(f'Subject {sub} {ch} {bl_event}-{event} Linear SVM'
                     '\nClassification Feature Importances '
                     f'({score.round(2)})')
        plot_image(fig, ax, img, data_dict, vmin=-0.05, vmax=0.05)
        fig.savefig(op.join(out_dir, out_fname + '_features.png'))
        plt.close(fig)
        fig, axes = plt.subplots(2, 2, figsize=(10, 10))
        fig.suptitle('Spectrograms Based on Classification Accuracy '
                     f'({score.round(2)})')
        pred = classifier.predict(X_test_pca)
        tp = np.where(np.logical_and(pred == y_test, y_test == 1))[0]
        fp = np.where(np.logical_and(pred != y_test, y_test == 1))[0]
        tn = np.where(np.logical_and(pred == y_test, y_test == 0))[0]
        fn = np.where(np.logical_and(pred != y_test, y_test == 0))[0]
        plot_image(fig, axes[0, 0], np.median(epochs._data[tp], axis=0),
                   data_dict, vmin=-1, vmax=1, cbar=False, cmap='RdYlBu_r')
        axes[0, 0].set_title(f'True {event} ({len(tp)})')
        plot_image(fig, axes[0, 1], np.median(epochs._data[fp], axis=0),
                   data_dict, vmin=-1, vmax=1, cbar=False, cmap='RdYlBu_r')
        axes[0, 1].set_title(f'False {event} ({len(fp)})')
        plot_image(fig, axes[1, 1], np.median(bl_epochs._data[tn], axis=0),
                   data_dict, vmin=-1, vmax=1, cbar=False, cmap='RdYlBu_r')
        axes[1, 1].set_title(f'True {bl_event} ({len(tn)})')
        plot_image(fig, axes[1, 0], np.median(bl_epochs._data[fn], axis=0),
                   data_dict, vmin=-1, vmax=1, cbar=False, cmap='RdYlBu_r')
        axes[1, 0].set_title(f'False {bl_event} ({len(fn)})')
        for ax in axes.flatten()[1:]:
            ax.set_yticks([])
            ax.set_ylabel('')
            ax.set_xticks([])
            ax.set_xlabel('')
        fig.savefig(op.join(out_dir, out_fname + '_comparison.png'))
        plt.close(fig)
        weight = (score - 0.5) * 2
        feature_map = img * weight if \
            feature_map is None else feature_map + img * weight
        n_show = min([tp.size, fp.size, tn.size, fn.size])
        n_col = int(n_show ** 0.5)
        for name, idx in {f'True {event}': tp, f'False {event}': fp,
                          f'True {bl_event}': tn,
                          f'False {bl_event}': fn}.items():
            fig, axes = plt.subplots(n_col, n_show // n_col + 1,
                                     figsize=(15, 15))
            fig.suptitle(f'{name} ({len(idx)})')
            for i, ax in zip(np.random.choice(idx, n_show, replace=False),
                             axes.flatten()):
                epo = epochs._data[i] if y_test[i] else bl_epochs._data[i]
                plot_image(fig, ax, epo, data_dict, vmin=-5, vmax=5,
                           cmap='RdYlBu_r', cbar=False)
            for ax in axes.flatten():
                ax.axis('off')
            name_str = name.replace(' ', '_').lower()
            fig.savefig(op.join(out_dir, out_fname + f'_{name_str}.png'),
                        dpi=300)
            plt.close(fig)

np.savez_compressed(op.join(out_dir, 'scores.npz'), **scores)
np.savez_compressed(op.join(out_dir, 'imgs.npz'), **imgs)

# plot img over all channels, weighted
feature_map /= len(data)
fig, ax = plt.subplots()
fig.suptitle(f'{bl_event}-{event} PCA+Linear SVM\nClassification '
             f'Feature Importances Weighted')
plot_image(fig, ax, feature_map, data_dict, vmin=feature_map.min(),
           vmax=feature_map.max())
fig.savefig(op.join(out_dir, f'{bl_event}-{event}'
                    '_features.png').replace(' ', '_'), dpi=300)
plt.close(fig)
