import sys
import os
import os.path as op
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from mne.decoding import CSP

import mne

from utils import get_subjects, load_raw

from params import BIDS_ROOT as bids_root

# decoding-specific parameters
freqs = np.logspace(np.log(8), np.log(250), 50, base=np.e)
f_buffer = 0.5
windows = np.linspace(0, 2, 11)
windows = (windows[1:] + windows[:-1]) / 2  # take mean

subjects_dir = op.join(bids_root, 'derivatives')
out_dir = op.join(bids_root, 'derivatives', 'analysis_data')

if not op.isdir(out_dir):
    os.makedirs(out_dir)


for sub in get_subjects(__name__, sys.argv):
    out_fname = op.join(out_dir, f'sub-{sub}_csp_tf_scores.npz')
    if op.isfile(out_fname):
        continue
    raw = load_raw(sub)
    keep = np.array(pd.read_csv(op.join(
        subjects_dir, f'sub-{sub}', 'ieeg',
        f'sub-{sub}_reject_mask.tsv'), sep='\t')['keep'])
    events, event_id = mne.events_from_annotations(raw)
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
        bl_epochs = mne.Epochs(
            raw_filter, events[events[:, 2] == event_id['Fixation']][keep],
            detrend=1, baseline=None, preload=True,
            tmin=-2.5 - f_buffer, tmax=-0.5 + f_buffer, verbose=False)
        # extra 0.001 to match number of samples
        epochs = mne.Epochs(
            raw_filter, events[events[:, 2] == event_id['Response']][keep],
            detrend=1, tmin=-1 - f_buffer, tmax=1.001 + f_buffer,
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
    np.savez_compressed(out_fname, tf_scores)
