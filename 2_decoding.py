import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from mne.decoding import CSP

import mne
import mne_bids

from params import BIDS_ROOT as bids_root
from params import DATA_DIR as data_dir
from params import SUBJECTS as subjects
from params import TASK as task

# decoding-specific parameters
freqs = np.logspace(np.log(8), np.log(250), 50, base=np.e)
f_buffer = 0.5
windows = np.linspace(0, 2, 11)
windows = (windows[1:] + windows[:-1]) / 2  # take mean

subjects_dir = op.join(bids_root, 'derivatives')
plot_dir = op.join(data_dir, 'derivatives', 'decoding_plots')

if not op.isdir(plot_dir):
    os.makedirs(plot_dir)


for sub in subjects:
    path = mne_bids.BIDSPath(root=bids_root, subject=str(sub), task=task)
    raw = mne_bids.read_raw_bids(path)
    raw.pick_types(seeg=True)
    events, event_id = mne.events_from_annotations(raw)
    info = mne.io.read_info(op.join(
        subjects_dir, f'sub-{sub}', 'ieeg',
        f'sub-{sub}_task-{task}_info.fif'))
    raw.drop_channels([ch for ch in raw.ch_names if ch not in info.ch_names])
    raw.info = info
    # crop first to lessen amount of data, then load
    raw.crop(tmin=events[:, 0].min() / raw.info['sfreq'] - 5,
             tmax=events[:, 0].max() / raw.info['sfreq'] + 5)
    raw.load_data()
    raw.set_eeg_reference('average')
    # decoder analysis
    clf = make_pipeline(CSP(), LinearDiscriminantAnalysis())
    n_splits = 5  # how many folds to use for cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True)
    le = LabelEncoder()
    # do decoding
    tf_scores = np.zeros((freqs.size, windows.size))
    for i in tqdm(range(freqs.size)):
        fmin = freqs[0] - (freqs[1] - freqs[0]) / 2 if i == 0 else \
            np.mean([freqs[i], freqs[i - 1]])
        fmax = freqs[-1] + (freqs[-1] - freqs[-2]) / 2 if \
            i == freqs.size - 1 else np.mean([freqs[i], freqs[i + 1]])
        w_size = 7. / ((fmax + fmin) / 2.)  # in seconds
        raw_filter = raw.copy().filter(
            fmin, fmax, n_jobs=1, fir_design='firwin',
            skip_by_annotation='edge', verbose=False)
        bl_epochs = mne.Epochs(raw_filter, events, event_id['Fixation'],
                               detrend=1, baseline=None, preload=True,
                               tmin=-2.5 - f_buffer, tmax=-0.5 - f_buffer,
                               verbose=False)
        # extra 0.001 to match number of samples
        epochs = mne.Epochs(
            raw_filter, events, event_id['Response'], detrend=1,
            tmin=-1 - f_buffer, tmax=1.001 + f_buffer,
            baseline=None, preload=True, verbose=False)
        y = le.fit_transform(np.concatenate([bl_epochs.events[:, 2],
                                             epochs.events[:, 2]]))
        for t, w_time in tqdm(enumerate(windows)):
            bl_tmin = bl_epochs.tmin + f_buffer + w_time - w_size / 2
            bl_X = bl_epochs.copy().crop(
                bl_tmin, bl_tmin + w_size).get_data()
            tmin = epochs.tmin + f_buffer + w_time - w_size / 2
            X = epochs.copy().crop(
                tmin, tmin + w_size).get_data()
            X_all = np.concatenate([bl_X, X], axis=0)
            # some random inital conditions fail to converge
            these_scores = np.array([])
            while these_scores.size < 5:
                np.random.seed(datetime.now().microsecond)
                scores = cross_val_score(
                    estimator=clf, X=X_all, y=y, scoring='roc_auc',
                    cv=cv, n_jobs=1)
                these_scores = np.concatenate(
                    [these_scores, scores[~np.isnan(scores)]])
            tf_scores[i, t] = these_scores[:n_splits].mean()
    av_tfr = mne.time_frequency.AverageTFR(
        mne.create_info(['freq'], raw.info['sfreq']), tf_scores[np.newaxis, :],
        windows, freqs, 1)
    chance = np.mean(y)  # set chance level to white in the plot
    fig = av_tfr.plot([0], vmin=chance, cmap=plt.cm.Reds, show=False)[0]
    fig.suptitle('Fixation-Response Decoding Scores', fontsize=24)
    ax = fig.gca()
    ax.set_xticks([0, 0.5, 1, 1.5, 2])
    ax.set_xticklabels([-1, -0.5, 0, 0.5, 1], fontsize=24)
    ax.set_xlabel('Time (s)', fontsize=32)
    ax.set_yticks(freqs[::4].round())
    ax.set_yticklabels(freqs[::4].round().astype(int), fontsize=20)
    ax.set_ylabel('Frequency (Hz)', fontsize=32)
    fig.savefig(op.join(plot_dir, f'sub-{sub}_csp.png'))
    np.savez_compressed(op.join(plot_dir, f'sub-{sub}_csp_tf_scores.npz'),
                        tf_scores)