import sys
import os
import os.path as op
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import stats
from scipy.ndimage import gaussian_filter1d
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA

import mne

from utils import get_subjects, load_raw, compute_tfr, plot_confusion_matrix

from params import PLOT_DIR as plot_dir
from params import BIDS_ROOT as bids_root
from params import SUBJECTS as subjects
from params import ALPHA as alpha


subjects_dir = op.join(bids_root, 'derivatives')
out_dir = op.join(bids_root, 'derivatives', 'analysis_data')
spec_dir = op.join(plot_dir, 'derivatives', 'spectrogram_plots')

for this_dir in (out_dir, spec_dir):
    if not op.isdir(this_dir):
        os.makedirs(this_dir)

# %%
# First, optimize n_components parameter using a test-train-validation split

n_components_fname = op.join(out_dir, 'n_components_dist.npz')
if not op.isfile(n_components_fname):  # hyperparameter search
    n_components_dist = dict()
    rng = np.random.default_rng(seed=33)
    for sub in subjects:
        n_epochs = None  # check that the number of epochs is the same
        raw = load_raw(sub)
        keep = np.array(pd.read_csv(op.join(
            subjects_dir, f'sub-{sub}', 'ieeg',
            f'sub-{sub}_reject_mask.tsv'), sep='\t')['keep'])
        # use filtered raw for evoked
        raw_filtered = raw.copy().filter(l_freq=0.1, h_freq=40)
        # compute power, do manually for each channel to speed things up
        # use 5 random channels from each subject to do hyperparameter search
        idxs = rng.choice(np.arange(len(raw.ch_names)), 5)
        for i in idxs:
            ch = raw.ch_names[i]
            elec_name = ''.join([letter for letter in ch if
                                 not letter.isdigit()]).rstrip()
            number = ''.join([letter for letter in ch if
                              letter.isdigit()]).rstrip()
            name_str = f'sub-{sub}_ch-{elec_name}{number}'
            print(f'Computing {ch}')
            tfr_data = compute_tfr(raw, i, raw_filtered, keep)
            # compare baseline to event as well as null to baseline
            X = np.concatenate([tfr_data['baseline']['data'],
                                tfr_data['event']['data']], axis=0)
            X = X.reshape(X.shape[0], -1).astype('float32')  # flatten features
            y = np.concatenate(
                [np.repeat(0, tfr_data['baseline']['data'].shape[0]),
                 np.repeat(1, tfr_data['event']['data'].shape[0])])
            # be sure not to use testing data
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=0.2, random_state=99)
            # hyperparameter PCA n_components using 5-fold cross validation
            kf = KFold(n_splits=5)  # no shuffle, deterministic, no seed
            n_comp_scores = dict()
            for train_i, validate_i in kf.split(X_train):
                print('Cross validation...')
                n_comp_check = np.arange(1, train_i.size - 1)
                for n_components in n_comp_check:
                    pca = PCA(n_components=n_components,
                              svd_solver='randomized', whiten=True).fit(
                        X_train[train_i])
                    X_train_v_pca = pca.transform(X_train[train_i])
                    X_validate_pca = pca.transform(X_train[validate_i])
                    classifier = SVC(kernel='linear', random_state=99)
                    classifier.fit(X_train_v_pca, y_train[train_i])
                    score = classifier.score(
                        X_validate_pca, y_train[validate_i])
                    if n_components in n_comp_scores:
                        n_comp_scores[n_components].append(score)
                    else:
                        n_comp_scores[n_components] = [score]
            n_components_dist[name_str] = \
                {n_components: np.median(scores) for n_components, scores
                 in n_comp_scores.items()}
    np.savez_compressed(n_components_fname,
                        n_components_dist=n_components_dist)


with np.load(n_components_fname, allow_pickle=True) as n_components_data:
    n_components_dist = n_components_data['n_components_dist'].item()

# median subtract scores to compare then take the median of that
# for each component, smooth with gaussian kernal
n_comp_grand_median = dict()
for name in n_components_dist:
    score_med = np.median([n_components_dist[name][n_comp]
                           for n_comp in n_components_dist[name]])
    for n_comp in n_components_dist[name]:
        score = n_components_dist[name][n_comp]
        if n_comp in n_comp_grand_median:
            n_comp_grand_median[n_comp].append(score - score_med)
        else:
            n_comp_grand_median[n_comp] = [score - score_med]
n_comp_grand_median = {n_comp: np.median(scores) for n_comp, scores in
                       n_comp_grand_median.items()}
n_comps = list(n_comp_grand_median.keys())
scores = np.array([n_comp_grand_median[n_comp] for n_comp in n_comps])
scores_smooth = gaussian_filter1d(scores, sigma=5)
n_components_use = n_comps[np.argmax(scores_smooth)]


n_comp_plot_fname = op.join(plot_dir, 'derivatives', 'plots',
                            'n_comp_plot.png')
if not op.isfile(n_comp_plot_fname):
    fig, axes = plt.subplots(1, 3, figsize=(10, 6))
    axes[0].set_title('Raw Cross-Validation Scores by Channel')
    axes[0].set_ylabel('Score')
    for name in n_components_dist:
        n_comps = list(n_components_dist[name])
        axes[0].plot(n_comps, [n_components_dist[name][n_comp] for
                               n_comp in n_comps], alpha=0.5)
    axes[1].set_title('Median Subtracted Scores')
    axes[1].set_xlabel('Number of Principal Components')
    for name in n_components_dist:
        n_comps = list(n_components_dist[name])
        score_med = np.median([n_components_dist[name][n_comp]
                               for n_comp in n_components_dist[name]])
        axes[1].plot(n_comps, [n_components_dist[name][n_comp] - score_med
                               for n_comp in n_comps], alpha=0.5)
    axes[2].set_title('Grand Median Score with Smoothing')
    n_comps = list(n_comp_grand_median.keys())
    axes[2].plot(n_comps, scores, label='raw')
    axes[2].plot(n_comps, scores_smooth, label='smoothed')
    fig.tight_layout()
    fig.savefig(n_comp_plot_fname)

# %%
# The main classification

pca_vars = dict(event=dict(), go_event=dict(), null=dict())
svm_coef = dict(event=dict(), go_event=dict(), null=dict())
# scores per electrode
scores = dict(event=dict(), go_event=dict(), null=dict())
# correlation coefficient images
images = dict(event=dict(), go_event=dict(), null=dict())

clusters = dict(event=dict(), go_event=dict(), null=dict())
threshold = stats.distributions.t.ppf(1 - alpha, len(subjects) - 1)

rng = np.random.default_rng(seed=33)

for sub in get_subjects(__name__, sys.argv):
    out_fname = op.join(out_dir, f'sub-{sub}_pca_svm_data.npz')
    if op.isfile(out_fname):
        continue
    n_epochs = None  # check that the number of epochs is the same
    raw = load_raw(sub)
    keep = np.array(pd.read_csv(op.join(
        subjects_dir, f'sub-{sub}', 'ieeg',
        f'sub-{sub}_reject_mask.tsv'), sep='\t')['keep'])
    # use filtered raw for evoked
    raw_filtered = raw.copy().filter(l_freq=0.1, h_freq=40)
    # compute power, do manually for each channel to speed things up
    for i, ch in enumerate(raw.ch_names):
        elec_name = ''.join([letter for letter in ch if
                             not letter.isdigit()]).rstrip()
        number = ''.join([letter for letter in ch if
                          letter.isdigit()]).rstrip()
        name_str = f'sub-{sub}_ch-{elec_name}{number}'
        print(str(np.round(100 * i / len(raw.ch_names), 2)) + '% done', ch)
        tfr_data = compute_tfr(raw, i, raw_filtered, keep)
        # compare events
        for (bl_event, event) in [('baseline', 'event'),
                                  ('baseline', 'go_event'),
                                  ('baseline', 'null')]:
            print(f'{bl_event}-{event} comparion')
            # cluster permutation statistics
            T_obs, ch_clusters, cluster_p_values, _ = \
                mne.stats.permutation_cluster_1samp_test(
                    tfr_data[event]['data'] - tfr_data[bl_event]['data'],
                    n_permutations=1024, threshold=threshold, out_type='mask')
            T_corrected = np.nan * np.ones_like(T_obs)
            cluster_count = 0
            for c, p_val in zip(ch_clusters, cluster_p_values):
                if p_val <= alpha:
                    T_corrected[c] = T_obs[c]
                    cluster_count += 1
            print(cluster_count, np.nanmin(T_corrected),
                  np.nanmax(T_corrected))
            clusters[event][name_str] = T_corrected
            # PCA SVM
            X = np.concatenate([tfr_data[bl_event]['data'],
                                tfr_data[event]['data']], axis=0)
            X = X.reshape(X.shape[0], -1).astype('float32')  # flatten features
            y = np.concatenate(
                [np.repeat(0, tfr_data[bl_event]['data'].shape[0]),
                 np.repeat(1, tfr_data[event]['data'].shape[0])])
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=0.2, random_state=99)
            pca = PCA(n_components=n_components_use,
                      svd_solver='randomized', whiten=True).fit(X_train)
            pca_vars[event][name_str] = pca.explained_variance_ratio_
            X_train_pca = pca.transform(X_train)
            X_test_pca = pca.transform(X_test)
            classifier = SVC(kernel='linear', random_state=99)
            classifier.fit(X_train_pca, y_train)
            score = classifier.score(X_test_pca, y_test)
            scores[event][name_str] = score
            if n_epochs is None:
                n_epochs = (y_train.size, y_test.size)
            else:
                assert n_epochs == (y_train.size, y_test.size)
            eigenvectors = pca.components_.reshape(
                (n_components_use, len(tfr_data[event]['freqs']),
                 tfr_data[event]['times'].size))
            image = np.sum(
                classifier.coef_[0][:, np.newaxis, np.newaxis] * eigenvectors,
                axis=0)
            images[event][name_str] = image
            svm_coef[event][name_str] = classifier.coef_[0]
            # diagnostic plots
            pred = classifier.predict(X_test_pca)
            tp = np.where(np.logical_and(pred == y_test, y_test == 1))[0]
            fp = np.where(np.logical_and(pred != y_test, y_test == 1))[0]
            tn = np.where(np.logical_and(pred == y_test, y_test == 0))[0]
            fn = np.where(np.logical_and(pred != y_test, y_test == 0))[0]
            try:
                plot_confusion_matrix(
                    sub, elec_name, number, tfr_data, ch, event, bl_event,
                    score, image, y_test, tp, fp, tn, fn)
            except Exception as e:
                print(e)  # error for no data in a category, fix later
    np.savez_compressed(out_fname, n_epochs=n_epochs, clusters=clusters,
                        n_components=n_components_use, pca_vars=pca_vars,
                        svm_coef=svm_coef, scores=scores, images=images)
