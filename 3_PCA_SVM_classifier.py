import sys
import os
import os.path as op
import numpy as np
import pandas as pd

from scipy import stats
from sklearn.model_selection import train_test_split
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
plot_dir = op.join(plot_dir, 'derivatives', 'spectrogram_plots')

for this_dir in (out_dir, plot_dir):
    if not op.isdir(this_dir):
        os.makedirs(this_dir)


# dict of info for each contact
pca_vars = dict(event=dict(), null=dict())
scores = dict(event=dict(), null=dict())  # scores per electrode
images = dict(event=dict(), null=dict())  # correlation coefficient images

clusters = dict()
threshold = stats.distributions.t.ppf(1 - alpha, len(subjects) - 1)

rng = np.random.default_rng(seed=33)
min_keep = np.inf
for sub in subjects:
    keep = np.array(pd.read_csv(op.join(
        subjects_dir, f'sub-{sub}', 'ieeg',
        f'sub-{sub}_reject_mask.tsv'), sep='\t')['keep'])
    min_keep = min([min_keep, keep.sum()])

for sub in get_subjects(__name__, sys.argv):
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
        # cluster permutation statistics
        T_obs, ch_clusters, cluster_p_values, _ = \
            mne.stats.permutation_cluster_1samp_test(
                tfr_data['event']['data'] - tfr_data['baseline']['data'],
                n_permutations=1024, threshold=threshold, out_type='mask')
        T_corrected = np.nan * np.ones_like(T_obs)
        cluster_count = 0
        for c, p_val in zip(ch_clusters, cluster_p_values):
            if p_val <= alpha:
                T_corrected[c] = T_obs[c]
                cluster_count += 1
        print(cluster_count, np.nanmin(T_corrected), np.nanmax(T_corrected))
        clusters[name_str] = T_corrected
        # compare baseline to event as well as null to baseline
        for (bl_event, event) in [('baseline', 'event'), ('baseline', 'null')]:
            shuffle_idx = np.arange(tfr_data[event]['data'].shape[0] * 2)
            rng.shuffle(shuffle_idx)
            X = np.concatenate([tfr_data[bl_event]['data'],
                                tfr_data[event]['data']], axis=0)
            X = X.reshape(X.shape[0], -1).astype('float32')  # flatten features
            y = np.concatenate(
                [np.repeat(0, tfr_data[event]['data'].shape[0]),
                 np.repeat(1, tfr_data[bl_event]['data'].shape[0])])
            X = X[shuffle_idx][:min_keep]
            y = y[shuffle_idx][:min_keep]
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=.2, random_state=99)
            pca = PCA(n_components=X_train.shape[0] - 1,
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
                (X_train.shape[0] - 1, len(tfr_data[event]['freqs']),
                 tfr_data[event]['times'].size))
            image = np.sum(
                classifier.coef_[0][:, np.newaxis, np.newaxis] * eigenvectors,
                axis=0)
            images[event][name_str] = image
            # diagnostic plots
            pred = classifier.predict(X_test_pca)
            tp = np.where(np.logical_and(pred == y_test, y_test == 1))[0]
            fp = np.where(np.logical_and(pred != y_test, y_test == 1))[0]
            tn = np.where(np.logical_and(pred == y_test, y_test == 0))[0]
            fn = np.where(np.logical_and(pred != y_test, y_test == 0))[0]
            plot_confusion_matrix(
                sub, elec_name, number, tfr_data, ch, event, bl_event,
                score, image, y_test, tp, fp, tn, fn)
    np.savez_compressed(
        op.join(out_dir, f'sub-{sub}_pca_svm_data.npz'),
        n_epochs=n_epochs, clusters=clusters,
        pca_vars=pca_vars, scores=scores, images=images)

'''
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
'''
