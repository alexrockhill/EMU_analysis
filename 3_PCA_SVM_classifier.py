import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA

import mne

from utils import load_raw

from params import PLOT_DIR as plot_dir
from params import BIDS_ROOT as bids_root
from params import SUBJECTS as subjects
from params import TASK as task
from params import FREQUENCIES as freqs
from params import EVENTS as event_dict
from params import ALPHA as alpha


def plot_evoked(raw):
    # plot evoked
    events, event_id = mne.events_from_annotations(raw)
    for name, (event, tmin, tmax) in event_dict.items():
        epochs = mne.Epochs(raw, events, event_id[event], preload=True,
                            tmin=tmin, tmax=tmax, detrend=1, baseline=None)
        fig = epochs.average().plot(show=False)
        fig.savefig(op.join(plot_dir, f'sub-{sub}_event-{name}_evoked.png'))
        plt.close(fig)
        fig = epochs.plot_psd(fmax=250, show=False)
        fig.savefig(op.join(plot_dir, f'sub-{sub}_event-{name}_psd.png'))
        plt.close(fig)


def compute_tfr(raw, i, raw_filtered):
    sfreq = raw.info['sfreq']
    events, event_id = mne.events_from_annotations(raw)
    raw_data = raw._data[i].reshape(1, 1, raw._data.shape[1])
    raw_tfr = mne.time_frequency.tfr_array_morlet(
        raw_data, sfreq, freqs, output='power', verbose=False)
    raw_tfr = np.concatenate(
        [raw_filtered._data[i].reshape(1, 1, 1, raw._data.shape[1]),
         raw_tfr], axis=2)  # add DC
    info = mne.create_info(['0'] + [f'{np.round(f, 2)}' for f in freqs],
                           sfreq, 'seeg')
    raw_tfr = mne.io.RawArray(raw_tfr.squeeze(), info, raw.first_samp,
                              verbose=False)
    # use time from beginning of the first event to end of the last event
    full_tfr = mne.Epochs(raw_tfr, events, event_id['Fixation'],
                          preload=True, tmin=-2.5, tmax=2, baseline=None,
                          verbose=False)
    tfr_data = dict()
    for name, (event, tmin, tmax) in event_dict.items():
        tfr = mne.Epochs(raw_tfr, events, event_id[event],
                         preload=True, tmin=tmin, tmax=tmax,
                         baseline=None, verbose=False)
        med = np.median(full_tfr._data, axis=2)[:, :, np.newaxis]
        std = np.std(full_tfr._data, axis=2)[:, :, np.newaxis]
        tfr._data = (tfr._data - med) / std
        tfr_evo = mne.time_frequency.AverageTFR(
            raw.info.copy().pick_channels([ch]),
            np.median(tfr._data, axis=0)[np.newaxis],
            tfr.times, [0] + list(freqs), nave=tfr._data.shape[0],
            verbose=False)
        fig = tfr_evo.plot(show=False, verbose=False)[0]
        fig.suptitle(f'{ch} {event} Total Power')
        basename = 'sub-{}_ch-{}_event-{}'.format(
            sub, ch.replace(' ', ''), name)
        fig.savefig(op.join(plot_dir, basename + '_spectrogram.png'))
        plt.close(fig)
        tfr_data[name] = dict(data=tfr._data, times=tfr.times,
                              sfreq=sfreq, freqs=[0] + list(freqs))
    return tfr_data


def plot_image(fig, ax, img, freqs, times, vmin=None, vmax=None,
               cmap='RdYlBu_r', cbar=True):  # helper function
    vmin = img.min() if vmin is None else vmin
    vmax = img.max() if vmax is None else vmax
    c = ax.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
    ax.invert_yaxis()
    ax.set_xlabel('Time (s)')
    ticks = np.linspace(0, times.size - 1, 5).round().astype(int)
    ax.set_xticks(np.linspace(0, times.size, 5))
    ax.set_xticklabels(times[ticks].round(2))
    ax.set_ylabel('Frequency (Hz)')
    ax.set_yticks(range(len(freqs)))
    ax.set_yticklabels([f'{f}        ' if i % 2 else f for i, f in
                        enumerate(np.array(freqs).round(
                        ).astype(int))], fontsize=8)
    if cbar:
        fig.colorbar(c)
    fig.tight_layout()
    return c


def plot_confusion_matrix(out_fname, tfr_data, tp, fp, tn, fn):
    # coefficients plot
    fig, ax = plt.subplots()
    fig.suptitle(f'Subject {sub} {ch} {event_dict[bl_event][0]}-'
                 f'{event_dict[event][0]} Linear SVM'
                 '\nClassification Feature Importances '
                 f'({score.round(2)})')
    plot_image(fig, ax, image, tfr_data[event]['freqs'],
               tfr_data[event]['times'], vmin=-0.05, vmax=0.05)
    fig.savefig(op.join(plot_dir, out_fname + '_features.png'))
    plt.close(fig)
    # spectrograms by classification plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f'{ch} {event_dict[event][0]} Spectrograms Based '
                 f'on Classification Accuracy ({score.round(2)})')
    plot_image(fig, axes[0, 0],
               np.median(tfr_data[event]['data'][tp], axis=0),
               tfr_data[event]['freqs'], tfr_data[event]['times'],
               vmin=-1, vmax=1, cbar=False, cmap='RdYlBu_r')
    axes[0, 0].set_title(
        f'True {event_dict[event][0].title()} ({len(tp)})')
    plot_image(fig, axes[0, 1],
               np.median(tfr_data[event]['data'][fp], axis=0),
               tfr_data[event]['freqs'], tfr_data[event]['times'],
               vmin=-1, vmax=1, cbar=False, cmap='RdYlBu_r')
    axes[0, 0].set_xlabel('')
    axes[0, 1].set_title(
        f'False {event_dict[event][0].title()} ({len(fp)})')
    plot_image(fig, axes[1, 1],
               np.median(tfr_data[bl_event]['data'][tn], axis=0),
               tfr_data[bl_event]['freqs'],
               tfr_data[bl_event]['times'],
               vmin=-1, vmax=1, cbar=False, cmap='RdYlBu_r')
    axes[0, 1].set_xlabel('')
    axes[1, 1].set_title(
        f'True {event_dict[bl_event][0].title()} ({len(tn)})')
    c = plot_image(fig, axes[1, 0],
                   np.median(tfr_data[bl_event]['data'][fn], axis=0),
                   tfr_data[bl_event]['freqs'],
                   tfr_data[bl_event]['times'],
                   vmin=-1, vmax=1, cbar=False, cmap='RdYlBu_r')
    axes[1, 0].set_title(
        f'False {event_dict[bl_event][0].title()} ({len(fn)})')
    fig.subplots_adjust(top=0.9, right=0.9, bottom=0.07, hspace=0.2)
    cax = fig.add_axes([0.915, 0.1, 0.01, 0.8])
    fig.colorbar(c, cax=cax)
    cax.set_ylabel(r'Power ($\mu$$V^{2}$)', labelpad=0)
    fig.savefig(op.join(plot_dir, out_fname + '_comparison.png'))
    plt.close(fig)
    # single random example spectrograms plot
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
            if y_test[i]:
                plot_image(fig, ax, tfr_data[event]['data'][i],
                           tfr_data[event]['freqs'],
                           tfr_data[event]['times'],
                           vmin=-5, vmax=5, cmap='RdYlBu_r',
                           cbar=False)
            else:
                plot_image(fig, ax, tfr_data[bl_event]['data'][i],
                           tfr_data[bl_event]['freqs'],
                           tfr_data[bl_event]['times'],
                           vmin=-5, vmax=5, cmap='RdYlBu_r',
                           cbar=False)
        for ax in axes.flatten():
            ax.axis('off')
        name_str = name.replace(' ', '_').lower()
        fig.tight_layout()
        fig.savefig(
            op.join(plot_dir, out_fname + f'_{name_str}.png'),
            dpi=300)
        plt.close(fig)


subjects_dir = op.join(bids_root, 'derivatives')
out_dir = op.join(bids_root, 'derivatives', 'analysis_data')
plot_dir = op.join(plot_dir, 'derivatives', 'spectrogram_plots')

for this_dir in (out_dir, plot_dir):
    if not op.isdir(this_dir):
        os.makedirs(this_dir)


# table of info for each contact
subject = list()
electrode_name = list()  # name of the electrode shaft
contact_number = list()  # number of contact

pca_vars = dict(event=list(), null=list())
scores = dict(event=list(), null=list())  # scores per electrode
images = dict(event=dict(), null=dict())  # correlation coefficient images

clusters = dict()
threshold = stats.distributions.t.ppf(1 - alpha, len(subjects) - 1)

n_epochs = dict()  # number of epochs per subject
for sub in subjects:
    raw = load_raw(bids_root, sub, task, subjects_dir)
    plot_evoked(raw)
    # use filtered raw for evoked
    raw_filtered = raw.copy().filter(l_freq=0.1, h_freq=40)
    # compute power, do manually for each channel to speed things up
    for i, ch in enumerate(raw.ch_names):
        subject.append(sub)
        elec_name = ''.join([letter for letter in ch if
                             not letter.isdigit()]).rstrip()
        number = ''.join([letter for letter in ch if
                          letter.isdigit()]).rstrip()
        electrode_name.append(elec_name)
        contact_number.append(number)
        print(str(np.round(100 * i / len(raw.ch_names), 2)) + '% done', ch)
        tfr_data = compute_tfr(raw, i, raw_filtered)
        # cluster permutation statistics
        T_obs, ch_clusters, cluster_p_values, _ = \
            mne.stats.permutation_cluster_1samp_test(
                tfr_data['event']['data'] - tfr_data['baseline']['data'],
                n_permutations=1024, threshold=threshold, out_type='mask')
        T_corrected = np.nan * np.ones_like(T_obs)
        for c, p_val in zip(ch_clusters, cluster_p_values):
            if p_val <= alpha:
                T_corrected[c] = T_obs[c]
        print(np.nanmin(T_corrected), np.nanmax(T_corrected))
        clusters[f'sub-{sub}_ch-{elec_name}{number}'] = T_corrected
        # compare baseline to event as well as null to baseline
        for (bl_event, event) in [('baseline', 'event'), ('baseline', 'null')]:
            X = np.concatenate([tfr_data[bl_event]['data'],
                                tfr_data[event]['data']], axis=0)
            X = X.reshape(X.shape[0], -1).astype('float32')  # flatten features
            y = np.concatenate(
                [np.repeat(0, tfr_data[event]['data'].shape[0]),
                 np.repeat(1, tfr_data[bl_event]['data'].shape[0])])
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=.2, random_state=99)
            pca = PCA(n_components=X_train.shape[0] - 1,
                      svd_solver='randomized', whiten=True).fit(X_train)
            pca_vars[event].append(pca.explained_variance_ratio_)
            X_train_pca = pca.transform(X_train)
            X_test_pca = pca.transform(X_test)
            classifier = SVC(kernel='linear', random_state=99)
            classifier.fit(X_train_pca, y_train)
            score = classifier.score(X_test_pca, y_test)
            scores[event].append(score)
            if str(sub) in n_epochs:
                assert n_epochs[str(sub)] == y_test.size
            else:
                n_epochs[str(sub)] = y_test.size
            eigenvectors = pca.components_.reshape(
                (X_train.shape[0] - 1, len(tfr_data[event]['freqs']),
                 tfr_data[event]['times'].size))
            image = np.sum(
                classifier.coef_[0][:, np.newaxis, np.newaxis] * eigenvectors,
                axis=0)
            images[event][f'sub-{sub}_ch-{elec_name}{number}'] = image
            # diagnostic plots
            pred = classifier.predict(X_test_pca)
            tp = np.where(np.logical_and(pred == y_test, y_test == 1))[0]
            fp = np.where(np.logical_and(pred != y_test, y_test == 1))[0]
            tn = np.where(np.logical_and(pred == y_test, y_test == 0))[0]
            fn = np.where(np.logical_and(pred != y_test, y_test == 0))[0]
            plot_confusion_matrix(
                f'sub-{sub}_ch-{elec_name}{number}_{bl_event}-{event}',
                tfr_data, tp, fp, tn, fn)


np.savez_compressed(op.join(out_dir, 'n_epochs.npz'), **n_epochs)
np.savez_compressed(op.join(out_dir, 'clusters.npz'), **clusters)
np.savez_compressed(op.join(out_dir, 'event_pca_vars.npz'),
                    **pca_vars['event'])
np.savez_compressed(op.join(out_dir, 'null_pca_vars.npz'),
                    **pca_vars['null'])
score_data = pd.DataFrame(dict(sub=subject, elec_name=electrode_name,
                               number=contact_number,
                               event_scores=scores['event'],
                               null_scores=scores['null']))
score_data.to_csv(op.join(out_dir, 'scores.tsv'), sep='\t', index=False)
np.savez_compressed(op.join(out_dir, 'event_images.npz'), **images['event'])
np.savez_compressed(op.join(out_dir, 'null_images.npz'), **images['null'])
