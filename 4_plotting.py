import os
import os.path as op
import numpy as np
import pandas as pd

import mne
from mne.gui._ieeg_locate_gui import _CMAP
import nibabel as nib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.colors import Normalize

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from scipy import stats

from utils import load_raw, compute_tfr, DESTRIEUX_DICT

from params import PLOT_DIR as plot_dir
from params import BIDS_ROOT as bids_root
from params import EXTENSIONS as exts
from params import EVENTS as event_dict
from params import SUBJECTS as subjects
from params import TASK as task
from params import TEMPLATE as template
from params import ATLASES as asegs
from params import ALPHA as alpha
from params import LEFT_HANDED_SUBJECTS as lh_sub
from params import FREQUENCIES as freqs
from params import EXCLUDE_CH as exclude_ch

freqs = np.array([0] + list(freqs))  # add evoked


def swarm(x, bins):  # plot helper function
    counts = np.ones((bins.size))
    y = np.zeros((len(x)))
    for i, this_x in enumerate(x):
        idx = np.where(this_x < bins)[0][0] - 1
        y[i] = counts[idx] // 2 if counts[idx] % 2 else -counts[idx] // 2
        counts[idx] += 1
    return y


fig_dir = op.join(plot_dir, 'derivatives', 'plots')
data_dir = op.join(bids_root, 'derivatives', 'analysis_data')

if not op.isdir(fig_dir):
    os.makedirs(fig_dir)


# get plotting information
subjects_dir = op.join(bids_root, 'derivatives')
brain_kwargs = dict(cortex='low_contrast', alpha=0.2, background='white',
                    subjects_dir=subjects_dir, units='m')
lut, colors = mne._freesurfer.read_freesurfer_lut()
cmap = plt.get_cmap('viridis')
template_trans = mne.coreg.estimate_head_mri_t(template, subjects_dir)

# get svm information
scores, clusters, images, pca_vars, svm_coef = \
    ({event: dict() for event in ('event', 'go_event', 'null')}
     for _ in range(5))
for sub in subjects:
    print(f'Loading subject {sub} data')
    with np.load(op.join(data_dir, f'sub-{sub}_pca_svm_data.npz'),
                 allow_pickle=True) as data:
        for event in ('event', 'go_event', 'null'):
            scores[event].update(data['scores'].item()[event])
            clusters[event].update(data['clusters'].item()[event])
            images[event].update(data['images'].item()[event])
            pca_vars[event].update(data['pca_vars'].item()[event])
            svm_coef[event].update(data['svm_coef'].item()[event])

# exclude epileptogenic contacts
for name in exclude_ch:
    for event in ('event', 'go_event', 'null'):
        if name not in scores[event]:
            print(f'{name} not found')
            continue
        scores[event].pop(name)
        clusters[event].pop(name)
        images[event].pop(name)
        pca_vars[event].pop(name)
        svm_coef[event].pop(name)

for event in ('event', 'go_event', 'null'):
    print('Event {} variance explained {}+/-{}'.format(
        event,
        np.mean(np.sum(np.array(list(pca_vars[event].values())), axis=1)),
        np.std(np.sum(np.array(list(pca_vars[event].values())), axis=1))))


spec_shape = images[list(images.keys())[0]].shape
times = np.linspace(-0.5, 0.5, spec_shape[1])


# compute significant indices pooled across subjects
sig_thresh = np.quantile(list(null_scores.values()), 1 - alpha)
not_sig = [name for name, score in scores.items()
           if score <= sig_thresh]
sig = [name for name, score in scores.items()
       if score > sig_thresh]

# compute null distribution thresholds per image
image_thresh = np.quantile(
    abs(np.array(list(null_images.values()))), 1 - alpha)

# feature map computation
feature_maps = np.zeros((3, 2) + spec_shape)
for name, image in images.items():
    ch_cluster = clusters[name]
    score = scores[name]
    if score > sig_thresh:
        feature_maps[0, 0] += abs(image) > image_thresh  # count
        feature_maps[1, 0] += image > image_thresh
        feature_maps[2, 0] += score * (abs(image) > image_thresh)
        feature_maps[0, 1] += ~np.isnan(ch_cluster)
        feature_maps[1, 1] += ~np.isnan(ch_cluster) * ch_cluster > 0
        feature_maps[2, 1] += score * ~np.isnan(ch_cluster)

# normalize
feature_maps[1, 0] /= feature_maps[0, 0]  # scale by count
feature_maps[2, 0] /= feature_maps[0, 0]  # scale by count
feature_maps[0, 0] /= feature_maps[0, 0].max()
feature_maps[1, 1] /= feature_maps[0, 1]  # scale by count
feature_maps[2, 1] /= feature_maps[0, 1]  # scale by count
feature_maps[0, 1] /= feature_maps[0, 1].max()

# time-frequency areas of interest
prop_thresh = 1 / 3
areas = {'Pre-Movement Beta': (1, 22, 37, -0.35, -0.05),
         'Delta': (0, 1, 4, -0.5, 0.25),
         'Event-Related Potential': (1, 0, 0, -0.5, 0.5),
         'Post-Movement High-Beta': (1, 27, 36, 0.05, 0.2),
         'Post-Movement Low-Beta': (1, 14, 23, 0.1, 0.25),
         'Post-Movement Gamma': (0, 43, 140, 0.1, 0.23),
         'Pre-Movement Alpha': (0, 7, 13, -0.25, 0)}

area_directions = {'Pre-Movement Beta': (-1,), 'Delta': (1,),
                   'Event-Related Potential': (-1, 1),
                   'Post-Movement High-Beta': (1,),
                   'Post-Movement Low-Beta': (1,),
                   'Post-Movement Gamma': (1,),
                   'Pre-Movement Alpha': (-1, 1)}


area_contacts = {area: dict() for area in areas}
for name, cluster in clusters.items():
    mask = ~np.isnan(cluster) * np.sign(cluster)
    for area, (fm_idx, fmin, fmax, tmin, tmax) in areas.items():
        fmin_idx = np.argmin(abs(freqs - fmin))
        fmax_idx = np.argmin(abs(freqs - fmax))
        tmin_idx = np.argmin(abs(times - tmin))
        tmax_idx = np.argmin(abs(times - tmax))
        this_area = mask[slice(fmin_idx, fmax_idx + 1),
                         slice(tmin_idx, tmax_idx + 1)]
        area_contacts[area][name] = np.nansum(this_area) / this_area.size


# channel positions in template and individual
ch_pos = {'template': dict(), 'individual': dict()}
template_trans = mne.coreg.estimate_head_mri_t(template, subjects_dir)
for sub in subjects:  # first, find associated labels
    # individual
    info = mne.io.read_info(op.join(
        subjects_dir, f'sub-{sub}', 'ieeg', f'sub-{sub}_task-{task}_info.fif'))
    montage = mne.channels.make_dig_montage(
        dict(zip(info.ch_names, [ch['loc'][:3] for ch in info['chs']])),
        coord_frame='head')
    trans = mne.coreg.estimate_head_mri_t(f'sub-{sub}', subjects_dir)
    montage.apply_trans(trans)
    pos = montage.get_positions()['ch_pos']
    for ch_name, this_pos in pos.items():
        ch_name = ch_name.replace(' ', '')
        ch_pos['individual'][f'sub-{sub}_ch-{ch_name}'] = this_pos

    # template
    info = mne.io.read_info(op.join(
        subjects_dir, f'sub-{sub}', 'ieeg',
        f'sub-{sub}_template-{template}_task-{task}_info.fif'))
    montage = mne.channels.make_dig_montage(
        dict(zip(info.ch_names, [ch['loc'][:3] for ch in info['chs']])),
        coord_frame='head')
    montage.apply_trans(template_trans)
    pos = montage.get_positions()['ch_pos']
    for ch_name, this_pos in pos.items():
        ch_name = ch_name.replace(' ', '')
        ch_pos['template'][f'sub-{sub}_ch-{ch_name}'] = this_pos


# channel labels in individual space
ch_labels = {aseg: dict() for aseg in asegs}
for sub in subjects:  # first, find associated labels
    info = mne.io.read_info(op.join(
        subjects_dir, f'sub-{sub}', 'ieeg', f'sub-{sub}_task-{task}_info.fif'))
    trans = mne.coreg.estimate_head_mri_t(f'sub-{sub}', subjects_dir)
    montage = mne.channels.make_dig_montage(
        dict(zip(info.ch_names, [ch['loc'][:3] for ch in info['chs']])),
        coord_frame='head')
    montage.apply_trans(trans)
    for aseg in asegs:
        sub_labels = mne.get_montage_volume_labels(
            montage, f'sub-{sub}', subjects_dir=subjects_dir,
            aseg=aseg, dist=2)[0]
        for ch_name, labels in sub_labels.items():
            ch_name = ch_name.replace(' ', '')
            ch_labels[aseg][f'sub-{sub}_ch-{ch_name}'] = labels


def format_label_dk(label, combine_hemi=False, cortex=True):
    label = label.lower()
    # add spaces
    for kw in ('middle', 'inferior', 'superior', 'isthmus', 'temporal',
               'caudal', 'pars', 'rostral', 'medial', 'anterior',
               'frontal', 'occipital', 'lateral'):
        label = label.replace(kw, kw + ' ')
    label = label.replace('bankssts', 'banks of superior temporal sulcus')
    if 'ctx-' in label:
        label = label.replace('ctx-', '') + (' Cortex' if cortex else '')
    if combine_hemi:
        label = label.replace('lh-', '').replace('rh-', '').replace(
            'left-', '').replace('right-', '')
    else:
        if 'lh-' in label or 'left-' in label:
            label = 'Left ' + label.replace('lh-', '').replace('left-', '')
        if 'rh-' in label or 'right-' in label:
            label = 'Right ' + label.replace('rh-', '').replace('right-', '')
    return label.replace('-', ' ').title().strip()


def format_label_destrieux(label, combine_hemi=False, cortex=True):
    return DESTRIEUX_DICT[f'ctx_{label}'] if f'ctx_{label}' in DESTRIEUX_DICT \
        else DESTRIEUX_DICT[label]


#########
# Plots #
#########

# %%
# Figure 1: Schematic

sub = 1
raw = load_raw(sub=sub)
events, event_id = mne.events_from_annotations(raw)
raw_filtered = raw.copy().filter(l_freq=0.1, h_freq=40)
n_channels = 20
n_samples = int(10 * raw.info['sfreq'])
start_sample = events[99, 0]  # response event
i = raw.ch_names.index('LPM 1')
raw_data = 1e3 * raw._data[i:i + n_channels,
                           start_sample: start_sample + n_samples]
raw_data -= raw_data.mean(axis=1)[:, None]

keep = np.array(pd.read_csv(op.join(
    subjects_dir, f'sub-{sub}', 'ieeg',
    f'sub-{sub}_reject_mask.tsv'), sep='\t')['keep'])

event, tmin, tmax = event_dict['event']
epochs = mne.Epochs(
    raw, events[events[:, 2] == event_id[event]][keep],
    preload=True, tmin=tmin, tmax=tmax, detrend=1, baseline=None)
evoked = epochs.average()

n_epochs = 10
epochs_data = 1e4 * epochs.get_data(picks=['LPM 1'])[:n_epochs, 0]

tfr_data = compute_tfr(raw, i, raw_filtered, keep)

X = np.concatenate([tfr_data['baseline']['data'],
                    tfr_data['event']['data']], axis=0)
X = X.reshape(X.shape[0], -1).astype('float32')  # flatten features
y = np.concatenate(
    [np.repeat(0, tfr_data['baseline']['data'].shape[0]),
     np.repeat(1, tfr_data['event']['data'].shape[0])])
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, random_state=99)
pca = PCA(n_components=50,
          svd_solver='randomized', whiten=True).fit(X_train)
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)

classifier = SVC(kernel='linear', random_state=99)
classifier.fit(X_train_pca, y_train)
score = classifier.score(X_test_pca, y_test)

eigenvectors = pca.components_.reshape(
    (50, len(tfr_data['event']['freqs']),
     tfr_data['event']['times'].size))
image = np.sum(
    classifier.coef_[0][:, np.newaxis, np.newaxis] * eigenvectors,
    axis=0)

pred = classifier.predict(X_test_pca)
tp = np.where(np.logical_and(pred == y_test, y_test == 1))[0]
fp = np.where(np.logical_and(pred != y_test, y_test == 1))[0]
tn = np.where(np.logical_and(pred == y_test, y_test == 0))[0]
fn = np.where(np.logical_and(pred != y_test, y_test == 0))[0]

sr = 800 / 1200  # screen ratio

x_min, x_max = X_test_pca[:, 0].min() * 1.1, X_test_pca[:, 0].max() * 1.1
y_min, y_max = X_test_pca[:, 1].min() * 1.1, X_test_pca[:, 1].max() * 1.1
XX, YY = np.meshgrid(np.linspace(x_min, x_max, 1000),
                     np.linspace(y_min, y_max, 1000))
XY = np.zeros((XX.size, 50))  # n_components == 50
XY[:, 0] = XX.ravel()
XY[:, 1] = YY.ravel()
ZZ = classifier.decision_function(XY)
ZZ = ZZ.reshape(XX.shape)

fig, (ax, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8),
                                   gridspec_kw={'height_ratios': [1, 2, 1]})
fig.text(0.03, 0.97, 'a', fontsize=18)
ax.axis('off')
# fixation 700 + blank 700 + go 1200 + iti 4000 = 6600
ax.axis([-0.02, 6.62, -1, 1])
# main experimental design
ax.plot([0, 0, 0, 6.6, 6.6, 6.6], [0.2, -0.2, 0, 0, -0.2, 0.2], color='black')
# fixation
for t in (0.3, 0.4, 0.5, 0.6, 0.7):
    ax.plot([t, t], [-0.2, 0.2], color=(0.5, 0.5, 0.5))
ax.plot([0, 0.35, 0.7], [0.2, 0.35, 0.2], color=(0.5, 0.5, 0.5))  # zoom
ax.fill([0, 0.7, 0.7, 0, 0], 0.37 + np.array([0, 0, 0.7 * sr, 0.7 * sr, 0]),
        color=(0, 0, 0))
ax.fill([0.31, 0.39, 0.39, 0.31, 0.31],
        0.37 + sr * np.array([0.31, 0.31, 0.39, 0.39, 0.31]),
        color=(0.996, 0.996, 0.996))
ax.text(0.35, 0.55 + 0.7 * sr, 'Fixation\n300-700 ms jittered',
        va='center', ha='center', fontsize=8, color=(0.5, 0.5, 0.5))
# blank
for t in (0.3, 0.4, 0.5, 0.6, 0.7):
    ax.plot(0.7 + np.array([t, t]), [-0.2, 0.2], color=(0.7, 0.7, 0.7))
ax.plot([0.7, 1.05, 1.4], [-0.2, -0.35, -0.2], color=(0.7, 0.7, 0.7))  # zoom
ax.fill(0.7 + np.array([0, 0.7, 0.7, 0, 0]),
        -0.37 - np.array([0, 0, 0.7 * sr, 0.7 * sr, 0]), color=(0, 0, 0))
ax.text(1.05, -0.58 - 0.7 * sr, 'Blank\n300-700 ms jittered',
        va='center', ha='center', fontsize=8, color=(0.7, 0.7, 0.7))
# cue
ax.plot(1.4 + np.array([0.45, 0.45]), [-0.2, 0.2], color=(0.4, 0.4, 0.4))
ax.plot(1.4 + np.array([1.2, 1.2]), [-0.2, 0.2], color=(0.4, 0.4, 0.4))
ax.plot([1.4, 2.05, 2.6], [0.2, 0.5, 0.2], color=(0.4, 0.4, 0.4))  # zoom
ax.fill(1.75 + np.array([0, 0.7, 0.7, 0, 0]),
        0.53 + np.array([0, 0, 0.7 * sr, 0.7 * sr, 0]), color=(0, 0, 0))
ax.fill(1.75 + np.array([0.28, 0.42, 0.42, 0.28]),
        0.53 + sr * np.array([0.35, 0.47, 0.23, 0.35]),
        color=(0.996, 0.996, 0.996))
ax.text(2.5, 0.75, 'Cue\n1.4 or 4 x\npractice RT',
        va='center', ha='left', fontsize=8, color=(0.4, 0.4, 0.4))
# inter-trial interval
ax.plot([2.6, 4.6, 6.6], [0.2, 0.5, 0.2], color=(0.3, 0.3, 0.3))  # zoom
ax.fill(4.25 + np.array([0, 0.7, 0.7, 0, 0]),
        0.53 + np.array([0, 0, 0.7 * sr, 0.7 * sr, 0]), color=(0, 0, 0))
ax.text(5, 0.75, 'Inter-trial inveral\n4000 ms',
        va='center', ha='left', fontsize=8, color=(0.3, 0.3, 0.3))
# analysis markers
rt = 0.324
ax.plot(1.4 + np.array([rt, rt]), [-0.2, 0.2], color='red')
ax.fill_between([1.4 + rt - 0.5, 1.4 + rt + 0.5], -0.2, 0.2,
                color='red', alpha=0.25)
ax.plot([1.32, 1.8, 1.4 + rt + 0.5], [-0.27, -0.38, -0.22],
        color='red', alpha=0.25)
ax.text(2.3, -0.72, 'Response Epoch\n-500 to 500 ms\nrelative to\nresponse',
        va='center', ha='center', fontsize=8, color='red', alpha=0.5)
ax.fill_between([5.1, 6.1], -0.2, 0.2, color='blue', alpha=0.25)
ax.plot([5.13, 5.7, 6.07], [-0.22, -0.38, -0.22], color='blue', alpha=0.25)
ax.text(5.7, -0.72,
        'Baseline Epoch\n-1500 to -500 ms\nrelative to\nend of trial',
        va='center', ha='center', fontsize=8, color='blue', alpha=0.5)
ax.fill_between([4.1, 5.1], -0.2, 0.2, color='green', alpha=0.25)
ax.plot([4.13, 4.5, 5.07], [-0.22, -0.68, -0.22], color='green', alpha=0.25)
ax.text(4, -0.85, 'Null Epoch\n-2500 to -1500 ms\nrelative to end of trial',
        va='center', ha='center', fontsize=8, color='green', alpha=0.5)

ax = ax2
fig.text(0.03, 0.75, 'b', fontsize=18)
ax.axis('off')
ax.set_xlim([-0.1, 13.1])
ax.set_ylim([-2.1, 22.1])

# raw plot
ax.plot(np.linspace(0, 3, raw_data.shape[1]),
        (raw_data + np.linspace(10, 20, n_channels)[:, None]).T,
        color='black', linewidth=0.25)
ax.text(1.5, 21, 'Example Subject\nsEEG Recording', ha='center')
ax.text(1.5, 9.25, '...', ha='center', fontsize=32, color='gray')
ax.text(-0.5, 15, 'Amplitude (mV)', va='center', rotation=90)
ax.text(1.5, 8, 'Time (s)', ha='center', fontsize=8)
ax.plot([0, 3, 3, 0, 0], [20.5, 20.5, 19.5, 19.5, 20.5],
        color='red', linewidth=0.5)
ax.text(3.7, 20.5, 'Epoch', ha='center', fontsize=8)
for offset in np.linspace(10, 20, n_channels):
    ax.fill([3.25, 4, 4, 4.25, 4, 4, 3.25, 3.25],
            offset + np.array([0.1, 0.1, 0.15, 0, -0.15, -0.1, -0.1, 0.1]),
            color='tab:red' if offset == 20 else 'tab:blue')

'''
# evoked plot
ax.fill([3.45, 3.45, 3.4, 3.5, 3.6, 3.55, 3.55, 3.45],
        [9.5, 8.5, 8.5, 8, 8.5, 8.5, 9.5, 9.5], color='tab:green')

ax.plot(np.linspace(1.5, 4.5, evoked.data.shape[1]),
        7.5e4 * evoked.data.T + 3.5, color='black', linewidth=0.25)
ax.text(3, 6.5, 'Evoked Plot', ha='center')
ax.text(3, -3.5, 'Time Relative\nto Key Response', ha='center', color='r')
ax.plot([3, 3], [-1, 6], color='r')'''

# epochs plot
ax.plot(np.linspace(4.5, 7.5, epochs_data.shape[1]),
        (epochs_data + np.linspace(13, 20, n_epochs)[:, None]).T,
        color='black', linewidth=0.25)
ax.text(6, 21, 'Epochs for an\nExample Channel', ha='center', fontsize=8)
ax.text(4.65, 22.3, '1', ha='center', fontsize=8)
ax.scatter([4.65], [22.6], marker='o', s=120, clip_on=False,
           edgecolors='black', facecolors='none')
ax.plot([6, 6], [12.75, 20.5], color='r')
ax.text(6, 12, '...', ha='center', fontsize=32, color='gray')
ax.text(6, 10.5, 'Average', ha='center')
ax.plot(np.linspace(4.5, 7.5, epochs_data.shape[1]),
        epochs_data.mean(axis=0) + 10, color='black', linewidth=0.25)
ax.text(6, 7.5, 'Time Relative\nto Key Response',
        fontsize=8, ha='center', color='r')
ax.plot([6, 6], [9.75, 10.25], color='r')

ax.text(8.25, 20.5, 'TFR', ha='center', va='center', fontsize=8)
for offset in np.linspace(13, 20, n_epochs):
    ax.fill(8 + np.array([0, 0.5, 0.5, 0.75, 0.5, 0.5, 0, 0]),
            offset + np.array([0.1, 0.1, 0.15, 0, -0.15, -0.1, -0.1, 0.1]),
            color='tab:blue')

# spectrogram
ax.text(11, 20.8, 'Example Training\nSpectrograms', ha='center')
ax.text(9, 22.3, '2', ha='center', fontsize=8)
ax.scatter([9], [22.6], marker='o', s=120, clip_on=False,
           edgecolors='black', facecolors='none')
for i in range(10):
    x0, x1 = 9.75 - i / 10, 12.75 - i / 10
    y0, y1 = 14 - i / 10, 20 - i / 10
    ax.imshow(tfr_data['event']['data'][9 - i][::-1],
              extent=(x0, x1, y0, y1), zorder=i,
              aspect='auto', cmap='viridis')
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
            color='black', linewidth=0.5, zorder=i)
ax.text(13, 16, 'Frequency (Hz)', va='center', rotation=90, fontsize=8)
ax.text(10.25, 10, 'PCA', ha='center')
ax.fill([9.75, 9.75, 9.25, 10.25, 11.25, 10.75, 10.75, 9.75],
        [11.5, 10, 10, 9, 10, 10, 11.5, 11.5], color='tab:blue')

# pca
ax.text(10.25, 6.5, 'Component Weights\nfor each Training Epoch',
        ha='center', fontsize=10)
ax.text(7.9, 7.8, '3', ha='center', fontsize=8)
ax.scatter([7.9], [8.1], marker='o', s=120, clip_on=False,
           edgecolors='black', facecolors='none')
ax.plot(np.linspace(8.75, 11.75, X_train_pca.shape[1]),
        (0.1 * X_train_pca[:n_epochs] +
         np.linspace(2, 6, n_epochs)[:, None]).T)
ax.text(10.25, 1, '...', ha='center', fontsize=32, color=(0.5, 0.5, 0.5))
ax.text(12, 4, 'Epochs', rotation=90, va='center')
ax.bar(np.linspace(8.75, 11.75, X_train_pca.shape[1]),
       20 * pca.explained_variance_ratio_,
       3 / X_train_pca.shape[1] * 0.75)
ax.text(8.4, 0.25, 'EV', rotation=90, fontsize=8)
ax.text(10.25, -1, 'Components', ha='center')
ax.text(8.2, 3, 'SVM', ha='center', va='center')
ax.fill([8.625, 7.875, 7.875, 7.625, 7.875, 7.875, 8.625, 8.625],
        [4, 4, 4.5, 3, 1.5, 2, 2, 4], color='tab:blue')

# svm
ax.text(6, 6, 'SVM Coefficients', ha='center')
ax.text(4, 6.1, '4', ha='center', fontsize=8)
ax.scatter([4], [6.4], marker='o', s=120, clip_on=False,
           edgecolors='black', facecolors='none')
ax.plot(np.linspace(4.5, 7.5, classifier.coef_[0].size),
        5.25 + classifier.coef_[0] / abs(classifier.coef_[0]).max())
ax.imshow(image[::-1], extent=(4.5, 7.5, -2, 4), aspect='auto', cmap='viridis')
ax.text(2.5, 4.75, 'Classify', ha='center', fontsize=8)
ax.fill(1.5 + 3.5 * np.array([0.75, 0.25, 0.25, 0, 0.25, 0.25, 0.75, 0.75]),
        4 + 2 * np.array([0.1, 0.1, 0.15, 0, -0.15, -0.1, -0.1, 0.1]),
        color='tab:blue')
for i in range(10):
    x0, x1 = 2.25 - i / 10, 4.25 - i / 10
    y0, y1 = -1 - i / 10, 3 - i / 10
    ax.imshow(tfr_data['event']['data'][-i][::-1],
              extent=(x0, x1, y0, y1), zorder=i,
              aspect='auto', cmap='viridis')
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0],
            color='black', linewidth=0.5, zorder=i)
ax.text(1.5, -4.25, 'Example Testing\nSpectrograms and Component Weights',
        ha='center', fontsize=8)
ax.plot(np.linspace(0, 1.25, X_test_pca.shape[1]),
        (0.1 * X_test_pca[:n_epochs] +
         np.linspace(-0.5, 3, n_epochs)[:, None]).T, linewidth=0.5)
ax.text(0.625, -1.5, '...', ha='center', fontsize=24, color=(0.5, 0.5, 0.5))

# decision boundary
ax.text(2, 6.1, '5', ha='center', fontsize=8)
ax.scatter([2], [6.4], marker='o', s=120, clip_on=False,
           edgecolors='black', facecolors='none')
ax.contourf((XX - x_min) / x_max / 1.5, (YY - y_min) / y_max * 1.5 + 4.25, ZZ,
            cmap='RdBu', levels=20)
ax.scatter((X_test_pca[y_test == 0, 0] - x_min) / x_max / 1.5,
           (X_test_pca[y_test == 0, 1] - y_min) / y_max * 1.5 + 4.25,
           marker='o', color='r', s=0.5)
ax.scatter((X_test_pca[y_test == 1, 0] - x_min) / x_max / 1.5,
           (X_test_pca[y_test == 1, 1] - y_min) / y_max * 1.5 + 4.25,
           marker='o', color='b', s=0.5)
'''
ax.text(0.5, 6.5, 'Confusion Matrix', ha='center')
ax.text(-0.25, 5.4, 'TP', ha='center', va='center', fontsize=8)
ax.text(0.25, 5.4, f'{tp.size}', ha='center', va='center', fontsize=8)
ax.text(1.25, 5.4, 'FP', ha='center', va='center', fontsize=8)
ax.text(0.75, 5.4, f'{fp.size}', ha='center', va='center', fontsize=8)
ax.text(-0.25, 4.2, 'FN', ha='center', va='center', fontsize=8)
ax.text(0.25, 4.2, f'{fn.size}', ha='center', va='center', fontsize=8)
ax.text(1.25, 4.2, 'FN', ha='center', va='center', fontsize=8)
ax.text(0.75, 4.2, f'{tn.size}', ha='center', va='center', fontsize=8)
ax.plot([0, 1, 1, 0, 0],
        [3.5, 3.5, 6, 6, 3.5], color='black')
ax.plot([0, 1], [4.75, 4.75], color='black')
ax.plot([0.5, 0.5], [3.5, 6], color='black')
'''

# red outline
ax.plot([0, 4.4, 4.4], [7.5, 7.5, 22], color='tab:red')

# add eigenspectrograms
ax = ax3
fig.text(0.03, 0.285, 'c', fontsize=18)
ax.set_xlim([0, 3.5])
ax.set_ylim([0, 1])
for direction in ('right', 'top', 'bottom'):
    ax.spines[direction].set_visible(False)
ax.invert_yaxis()
ax.imshow(eigenvectors[0], extent=(0, 1, 0, 1), aspect='auto', cmap='viridis')
ax.imshow(eigenvectors[1], extent=(1.2, 2.2, 0, 1),
          aspect='auto', cmap='viridis')
ax.imshow(eigenvectors[2], extent=(2.4, 3.4, 0, 1),
          aspect='auto', cmap='viridis')
ax.set_xticks([0, 0.5, 1, 1.2, 1.7, 2.2, 2.4, 2.9, 3.4])
ax.set_xticklabels([-0.5, 0, 0.5] * 3)
ax.set_xlabel('Time (s)')
ax.set_yticks(np.linspace(1, 0, len(freqs)))
ax.set_yticklabels([f'{f}        ' if i % 2 else f for i, f in
                    enumerate(np.array(freqs).round(
                    ).astype(int))], fontsize=4)
ax.set_ylabel('Frequency (Hz)')
pos = ax.get_position()
ax.set_position((pos.x0, pos.y0 - 0.05, pos.width, pos.height))

fig.subplots_adjust(top=0.98, bottom=0.1, left=0.1, right=0.98)
for ext in exts:
    fig.savefig(op.join(fig_dir, f'schematic.{ext}'), dpi=300)

# %%
# Figure 2: Individual implant plots to show sampling

fig, axes = plt.subplots(len(subjects) // 2, 6, figsize=(12, 8))
axes = axes.reshape(len(subjects), 3)
for ax in axes.flatten():
    for direction in ('left', 'right', 'top', 'bottom'):
        ax.spines[direction].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.invert_yaxis()


axes[0, 0].set_title('Right front')
axes[0, 1].set_title('Top down')
axes[0, 2].set_title('Left front')
for i, sub in enumerate(subjects):
    axes[i, 0].set_ylabel(f'Subject {sub}')
    info = mne.io.read_info(op.join(subjects_dir, f'sub-{sub}', 'ieeg',
                                    f'sub-{sub}_task-{task}_info.fif'))
    trans = mne.coreg.estimate_head_mri_t(f'sub-{sub}', subjects_dir)
    brain = mne.viz.Brain(f'sub-{sub}', **brain_kwargs)
    groups = dict()
    for ch_name in info.ch_names:
        elec_name = ''.join([letter for letter in ch_name if
                             not letter.isdigit() and letter != ' '])
        groups[ch_name] = elec_name
    chs = {ch['ch_name']: mne.transforms.apply_trans(trans, ch['loc'][:3])
           for ch in info['chs']}
    for idx, group in enumerate(np.unique(list(groups.values()))):
        pos = np.array([chs[ch] for i, ch in enumerate(info.ch_names)
                        if groups[ch] == group])
        # first, the insertion will be the point farthest from the origin
        # brains are a longer posterior-anterior, scale for this (80%)
        insert_idx = np.argmax(np.linalg.norm(pos * np.array([1, 0.8, 1]),
                                              axis=1))
        # second, find the farthest point from the insertion
        target_idx = np.argmax(np.linalg.norm(pos[insert_idx] - pos, axis=1))
        # third, make a unit vector and to add to the insertion for the bolt
        elec_v = pos[insert_idx] - pos[target_idx]
        elec_v /= np.linalg.norm(elec_v)
        brain._renderer.tube(  # 30 mm outside head
            [pos[target_idx]], [pos[insert_idx] + elec_v * 0.03],
            radius=0.001, color=_CMAP(idx)[:3])[0]
        for (x, y, z) in pos:
            brain._renderer.sphere(center=(x, y, z), color=_CMAP(idx)[:3],
                                   scale=0.005)
    # will add above code to add_sensors eventually
    # brain.add_sensors(info, trans)
    brain.show_view(azimuth=60, elevation=100, distance=0.325)
    axes[i, 0].imshow(brain.screenshot())
    brain.show_view(azimuth=90, elevation=0, distance=0.36)
    axes[i, 1].imshow(brain.screenshot())
    brain.show_view(azimuth=120, elevation=100, distance=0.325)
    axes[i, 2].imshow(brain.screenshot())
    brain.close()


fig.subplots_adjust(left=0.03, right=1, top=0.95, bottom=0.03,
                    wspace=-0.3, hspace=0)
for ax in axes[::2].flatten():
    pos = ax.get_position()
    ax.set_position((pos.x0 - 0.02, pos.y0, pos.width, pos.height))


for ax in axes[1::2].flatten():
    pos = ax.get_position()
    ax.set_position((pos.x0 + 0.02, pos.y0, pos.width, pos.height))

for ext in exts:
    fig.savefig(op.join(fig_dir, f'coverage.{ext}'), dpi=300)

# %%
# Figure 3: histogram of classification accuracies
#
# Radial basis function scores not shown, almost exactly the same

binsize = 0.01
bins = np.linspace(binsize, 1, int(1 / binsize)) - binsize / 2
fig, ax = plt.subplots()

patches = ax.hist(list(scores.values()), bins=bins,
                  alpha=0.5, color='b')[2]
for i, left_bin in enumerate(bins[:-1]):
    if left_bin > sig_thresh:
        patches[i].set_facecolor('r')
ax.hist(list(null_scores.values()), bins=bins, alpha=0.5, color='gray',
        label='null')
y_bounds = ax.get_ylim()
ax.axvline(np.mean(list(scores.values())), *y_bounds, color='black')
ax.axvline(np.mean(list(null_scores.values())), *y_bounds, color='gray')
ax.set_xlim([0.25, 1])
ax.set_xlabel('Test Accuracy')
ax.set_ylabel('Count')
not_sig_patch = mpatches.Patch(color='b', alpha=0.5, label='not significant')
sig_patch = mpatches.Patch(color='r', alpha=0.5, label='significant')
ax.legend(handles=[not_sig_patch, sig_patch])
fig.suptitle('PCA Linear SVM Classification Accuracies')
for ext in exts:
    fig.savefig(op.join(fig_dir, f'score_hist.{ext}'), dpi=300)

print('Paired t-test p-value: {}'.format(
    stats.ttest_rel(list(scores.values()),
                    list(null_scores.values()))[1]))
print('Significant contacts {} / {}'.format(
    (np.array(list(scores.values())) > sig_thresh).sum(), len(scores)))

# %%
# Figure 4: Plots of electrodes with high classification accuracies

fig = plt.figure(figsize=(8, 6))
gs = fig.add_gridspec(3, 4)
axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(3)]
                 for i in range(3)])
cax = fig.add_subplot(gs[:2, 3])
cax2 = fig.add_subplot(gs[2, 3])
for ax in axes.flatten():
    ax.axis('off')
    ax.invert_yaxis()


# color contacts by accuracy
brain = mne.viz.Brain(template, **brain_kwargs)
norm = Normalize(vmin=0, vmax=1)
for name, score in scores.items():
    if score > sig_thresh:
        x, y, z = ch_pos['template'][name]
        brain._renderer.sphere(center=(x, y, z),
                               color=cmap(norm(score))[:3],
                               scale=0.005)


axes[0, 0].set_title('Right front')
axes[0, 1].set_title('Top down')
axes[0, 2].set_title('Left front')
brain.show_view(azimuth=60, elevation=100, distance=.3)
axes[0, 0].imshow(brain.screenshot())
brain.show_view(azimuth=90, elevation=0)
axes[0, 1].imshow(brain.screenshot())
brain.show_view(azimuth=120, elevation=100)
axes[0, 2].imshow(brain.screenshot())
brain.close()
fig.text(0.1, 0.85, 'a')

# get labels
ignore_keywords = ('unknown', '-vent', 'choroid-plexus', 'vessel',
                   'white-matter', 'wm-', 'cc_', 'cerebellum',
                   'brain-stem')

labels = dict()
for name, score in scores.items():
    these_labels = ch_labels[asegs[0]][name]  # use plotting atlas
    for label in these_labels:
        if any([kw in label.lower() for kw in ignore_keywords]):
            continue
        if label in labels:
            labels[label].append(score)
        else:
            labels[label] = [score]


label_names = list(labels.keys())
acc_colors = [cmap(norm(np.mean(labels[name]))) for name in label_names]

brain = mne.viz.Brain(template, **dict(brain_kwargs, alpha=0))
brain.add_volume_labels(aseg=asegs[0], labels=label_names,
                        colors=acc_colors, alpha=1, smooth=0.9)
brain.show_view(azimuth=60, elevation=100, distance=.3)
axes[1, 0].imshow(brain.screenshot())
brain.show_view(azimuth=90, elevation=0)
axes[1, 1].imshow(brain.screenshot())
brain.show_view(azimuth=120, elevation=100)
axes[1, 2].imshow(brain.screenshot())
brain.close()
fig.text(0.1, 0.6, 'b')

# colorbar
gradient = np.linspace(0, 1, 256)
gradient = np.repeat(gradient[:, np.newaxis], 256, axis=1)
cax.imshow(gradient, aspect='auto', cmap=cmap)
cax.set_xticks([])
cax.invert_yaxis()
cax.yaxis.tick_right()
cax.set_ylim(np.array([sig_thresh, 1]) * 256)
cax.set_yticks(np.array([sig_thresh, 0.75, 1]) * 256)
cax.set_yticklabels([np.round(sig_thresh, 2), 0.75, 1])
cax.yaxis.set_label_position('right')
cax.set_ylabel('Accuracy')

# plot counts of electrodes per area
counts = dict()
for these_labels in ch_labels[asegs[0]].values():
    for label in these_labels:
        if any([kw in label.lower() for kw in ignore_keywords]):
            continue
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1


density_colors = [cmap(min([counts[name] / 10, 1.])) for name in label_names]

brain = mne.viz.Brain(template, **dict(brain_kwargs, alpha=0))
brain.add_volume_labels(aseg=asegs[0], labels=label_names,
                        colors=density_colors, alpha=1, smooth=0.9)
brain.show_view(azimuth=60, elevation=100, distance=.3)
axes[2, 0].imshow(brain.screenshot())
brain.show_view(azimuth=90, elevation=0)
axes[2, 1].imshow(brain.screenshot())
brain.show_view(azimuth=120, elevation=100)
axes[2, 2].imshow(brain.screenshot())
brain.close()
fig.text(0.1, 0.33, 'c')

# count colorbar
gradient = np.linspace(0, 10, 256)
gradient = np.repeat(gradient[:, np.newaxis], 256, axis=1)
cax2.imshow(gradient, aspect='auto', cmap=cmap)
cax2.set_xticks([])
cax2.invert_yaxis()
cax2.yaxis.tick_right()
cax2.set_yticks(np.linspace(2, 10, 5) * 256 / 10)
cax2.set_yticklabels(['2', '4', '6', '8', '10+'])
cax2.yaxis.set_label_position('right')
cax2.set_ylabel('Contact Count')

fig.subplots_adjust(hspace=0)
pos = cax.get_position()
cax.set_position((pos.x0, 0.35, 0.05, 0.5))
pos = cax2.get_position()
cax2.set_position((pos.x0, 0.1, 0.05, 0.2))
for ext in exts:
    fig.savefig(op.join(fig_dir, f'high_accuracy.{ext}'), dpi=300)

# %%
# Figure 5: Accuracy by label region of interest

ignore_keywords = ('unknown', '-vent', 'choroid-plexus', 'vessel', 'cc_',
                   'wm', 'cerebellum')  # signal won't cross dura
labels = set([label for labels in ch_labels[asegs[1]].values()
              for label in labels
              if not any([kw in label.lower() for kw in ignore_keywords])])
label_scores = dict()
for name, score in scores.items():
    these_labels = ch_labels[asegs[1]][name]  # use table atlas
    for label in these_labels:
        if not any([kw in label.lower() for kw in ignore_keywords]):
            if label in label_scores:
                label_scores[label].append(score)
            else:
                label_scores[label] = [score]
labels = sorted(labels, key=lambda label: np.mean(label_scores[label]))

fig, ax = plt.subplots(figsize=(8, 12), facecolor='black')
fig.suptitle('Classification Accuracies by Label', color='w')

for idx, label in enumerate(labels):
    for lh in (True, False):
        for sig_label, names in {'sig': sig, 'not_sig': not_sig}.items():
            these_scores = \
                [scores[name] for name in names
                 if label in ch_labels[asegs[1]][name] and
                 (lh == (int(name.split('_')[0].replace(
                     'sub-', '')) in lh_sub))]
            color = colors[label][:3] / 255
            if color.mean() > 0.9:
                color *= 0.75  # gray out white
            # triangle if left hand used, hollow if not significant
            ax.scatter(these_scores, [idx] * len(these_scores),
                       color=color, marker='^' if lh else None,
                       facecolors=None if sig_label == 'sig' else 'none')


ax.axis([0.25, 1, -0.75, len(labels) - 0.25])
ax.set_yticks(range(len(label_scores)))
ax.set_yticklabels([format_label_dk(label) for label in labels])
for tick, label in zip(ax.get_yticklabels(), labels):
    color = colors[label][:3] / 255
    tick.set_color('w' if color.max() < 0.6 or
                   (color[2] > 0.6 and color.mean() < 0.5)
                   else 'black')  # blue is dark
    tick.set_fontsize(8)
    tick.set_path_effects([patheffects.withStroke(
        linewidth=5, foreground=color)])


for tick in ax.get_xticklabels():
    tick.set_color('w')


ax.set_xlabel('Classification Accuracy', color='w')
ax.set_ylabel('Anatomical Label', color='w')

# make legend
ax.text(0.27, len(labels) - 2, 'Right hand', va='center')
ax.scatter([0.5], [len(labels) - 2], color='black')
ax.text(0.27, len(labels) - 3.5, 'Left hand', va='center')
ax.scatter([0.5], [len(labels) - 3.5], marker='^', color='black')
ax.text(0.27, len(labels) - 5, 'Significant', va='center')
ax.scatter([0.5], [len(labels) - 5], color='black')
ax.text(0.27, len(labels) - 6.5, 'Not significant', va='center')
ax.scatter([0.5], [len(labels) - 6.5], facecolors='none', color='black')
ax.plot([0.26, 0.26, 0.52, 0.52, 0.26],
        len(labels) - np.array([1, 7.15, 7.15, 1, 1]), color='black')

fig.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.07)
for ext in exts:
    fig.savefig(op.join(fig_dir, f'label_accuracies.{ext}'),
                facecolor=fig.get_facecolor(), dpi=300)


# %%
# Figure 6: distribution of classification accuracies across
# subjects compared to CSP.

# decoding-specific parameters
csp_freqs = np.logspace(np.log(8), np.log(250), 50, base=np.e)
windows = np.linspace(0, 2, 11)
windows = (windows[1:] + windows[:-1]) / 2  # take mean

fig, axes = plt.subplots(len(subjects) // 2, 4, figsize=(12, 8))
fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.08,
                    hspace=0.2, wspace=0.3)
axes = axes.reshape(len(subjects), 2)
binsize = 0.005
bins = np.linspace(0, 1 - binsize, int(1 / binsize))
for i, sub in enumerate(subjects):
    ax, ax2 = axes[i]
    these_scores = [scores[name] for name in scores if
                    int(name.split('_')[0].replace('sub-', '')) == sub]
    these_sig = [score for score in these_scores if score > sig_thresh]
    these_not_sig = [score for score in these_scores if score <= sig_thresh]
    ax.violinplot(these_sig + these_not_sig, [0],
                  vert=False, showextrema=False)
    y = swarm(these_sig, bins=bins) / 50
    ax.scatter(these_sig, y, color='r', s=2, label='sig')
    y = swarm(these_not_sig, bins=bins) / 50
    ax.scatter(these_not_sig, y, color='b', s=2, label='not sig')
    ax.set_ylabel(r'$\bf{Subject' + r'\enspace' + str(sub) + '}$\nDensity')
    ax.axis([0.25, 1, -0.28, 0.28])
    # CSP plot
    tf_scores = np.load(op.join(
        data_dir, f'sub-{sub}_csp_tf_scores.npz'))['arr_0']
    info = mne.io.read_info(op.join(subjects_dir, f'sub-{sub}', 'ieeg',
                                    f'sub-{sub}_task-{task}_info.fif'))
    av_tfr = mne.time_frequency.AverageTFR(
        mne.create_info(['freq'], info['sfreq']), tf_scores[np.newaxis, :],
        windows, csp_freqs, 1)
    av_tfr.plot([0], vmin=0.5, vmax=1, cmap=plt.cm.Reds, show=False, axes=ax2,
                colorbar=i % 2 == 1)
    if i % 2 == 0:  # adjust for not having colorbar
        pos = ax2.get_position()
        ax2.set_position((pos.x0, pos.y0, pos.width * 0.85, pos.height))
    ax2.set_xticks([0, 0.5, 1, 1.5, 2])
    ax2.set_xticklabels([-1, -0.5, 0, 0.5, 1])
    ax2.set_yticks(csp_freqs[::6].round())
    ax2.set_yticklabels(csp_freqs[::6].round().astype(int))
    ax2.set_ylabel('Frequency (Hz)')
    ax2.set_xlabel('')
    if i < 2:
        ax.set_title('SVM Accuracies')
        ax2.set_title('CSP Decoding')
    if i == len(subjects) - 1:
        ax.legend(loc='lower right')
    if i > len(subjects) - 3:
        ax.set_xlabel('Test Accuracy')
        ax2.set_xlabel('Time (s)')

for ext in exts:
    fig.savefig(op.join(fig_dir, f'svm_csp_comparison.{ext}'), dpi=300)

# %%
# Figure 7: Contacts of interest

ignore_keywords = ('unknown', '-vent', 'choroid-plexus', 'vessel',
                   'white-matter', 'wm-')

#              pre-movement beta,       gamma,     alpha/post-movement beta
contacts_int = ['sub-1_ch-LPM1', 'sub-5_ch-RPLS4', 'sub-10_ch-LACING6']

views = [dict(azimuth=-170, elevation=85, distance=0.25),
         dict(azimuth=50, elevation=15, distance=0.25),
         dict(azimuth=-160, elevation=30, distance=0.25)]

fig, axes = plt.subplots(len(contacts_int), 2, figsize=(6, 8))
axes[-1, 0].set_xlabel('Time (s)')
for ax in axes[:, 1]:
    ax.axis('off')


for ax in axes[:, 0]:
    ax.set_xticks(np.linspace(0, spec_shape[1] - 1, 5))
    ax.set_xticklabels([-0.5, -0.25, 0, 0.25, 0.5])


for (ax, ax2), name, view in zip(axes, contacts_int, views):
    sub = name.split('_ch-')[0].replace('sub-', '')
    elec_name = ''.join([letter for letter in name.split('ch-')[1]
                         if not letter.isdigit()])
    title_name = name.replace('sub-', 'Subject ').replace(
        '_ch-', ' ').replace(elec_name, elec_name + ' ')
    score = scores[name]
    ax.set_title(f'{title_name}, Test Accuracy {np.round(score, 2)}')
    elec = [name2 for name2 in ch_pos['individual'] if
            f'sub-{sub}_' in name2 and elec_name in name2]
    labels = set([label for name2 in elec for label in
                  ch_labels[asegs[1]][name2] if
                  not any([kw in label.lower() for kw in ignore_keywords])])
    # spectrogram plot
    image = images[name]
    cluster = clusters[name]
    mask = ~np.isnan(cluster)
    X, Y = np.meshgrid(range(image.shape[1]), range(image.shape[0]))
    img = ax.imshow(image, aspect='auto', vmin=-0.05, vmax=0.05,
                    cmap='viridis')
    ax.contour(X, Y, mask, levels=[0.5], colors=['r'], linewidths=0.5)
    ax.set_yticks(range(len(freqs)))
    ax.set_yticklabels([f'{f}        ' if i % 2 else f for i, f in
                        enumerate(np.array(freqs).round(
                        ).astype(int))], fontsize=6)
    ax.set_ylabel('Frequency (Hz)')
    ax.invert_yaxis()
    fig.colorbar(img, ax=ax)
    # anatomy plot
    brain = mne.viz.Brain(f'sub-{sub}', **dict(brain_kwargs, alpha=0.25))
    for name2 in elec:
        brain._renderer.sphere(
            ch_pos['individual'][name2],
            'black' if name2 == name else 'gray', 0.005)
    brain.add_volume_labels(aseg=asegs[1], labels=labels,
                            alpha=0.25, legend=False, fill_hole_size=1)
    # focus on halfway
    brain.show_view(focalpoint=ch_pos['individual'][elec[len(elec) // 2]],
                    **view)
    ax2.imshow(brain.screenshot())
    for label in labels:  # empty plots for legend handling
        ax2.scatter([np.nan], [np.nan], color=colors[label][:3] / 255,
                    marker='s', label=format_label_dk(label))
    ax2.legend(loc='lower left', fontsize='xx-small')
    brain.close()

fig.tight_layout()
for ext in exts:
    fig.savefig(op.join(fig_dir, f'contacts_of_interest.{ext}'), dpi=300)

# %%
# Figure 8: Feature maps

fig, axes = plt.subplots(3, 2, figsize=(8, 8))
for idx, ((svm_map, cluster_map), (ax1, ax2)) in enumerate(
        zip(feature_maps, axes)):
    vmin = 0 if idx < 2 else 0.75  # sig_thresh
    c = ax1.imshow(svm_map, vmin=vmin, vmax=1, cmap='viridis', aspect='auto')
    fig.colorbar(c, ax=ax1)
    c = ax2.imshow(cluster_map, vmin=vmin, vmax=1,
                   cmap='viridis', aspect='auto')
    fig.colorbar(c, ax=ax2)
    for ax in (ax1, ax2):
        ax.set_xticks(np.linspace(0, spec_shape[1], 5))
        ax.set_xticklabels([-0.5, -0.25, 0, 0.25, 0.5])
        ax.invert_yaxis()
    ax1.set_ylabel('Frequency (Hz)')
    ax1.set_yticks(range(len(freqs)))
    ax1.set_yticklabels([f'{f}        ' if i % 2 else f for i, f in
                         enumerate(np.array(freqs).round(
                         ).astype(int))], fontsize=5)
    ax2.set_yticks([])
    if idx == 0:
        ax1.set_title('Proportion of\nSignificant Coefficients')
        ax2.set_title('Proportion of\nSignificant Clusters')
    elif idx == 1:
        ax1.set_title('Proportion of Positive\nSignificant Coefficients')
        ax2.set_title('Proportion of Positive\nSignificant Clusters')
    else:
        ax1.set_title('Average Accuracy of\nSignificant Coefficients')
        ax2.set_title('Average Accuracy of\nSignificant Clusters')

for ax in axes[-1]:
    ax.set_xlabel('Time (s)')

fig.text(0.04, 0.95, 'a', fontsize=24)
fig.text(0.52, 0.95, 'b', fontsize=24)
fig.text(0.04, 0.63, 'c', fontsize=24)
fig.text(0.52, 0.63, 'd', fontsize=24)
fig.text(0.04, 0.31, 'e', fontsize=24)
fig.text(0.52, 0.31, 'f', fontsize=24)

fig.tight_layout()
for ext in exts:
    fig.savefig(op.join(fig_dir, f'feature_map.{ext}'), dpi=300)

# %%
# Figure 9: Anatomical Locations of Significant Correlations Areas

fig, axes = plt.subplots(len(areas), 5, figsize=(6.5, 10))

axes[-1, 0].set_xlabel('Time (s)')
axes[-1, 1].set_xlabel('Proportion of Area')
for ax in axes[:-1, :2].flatten():
    ax.set_xticks([])


for ax in axes[:, 1]:
    ax.set_yticks([])


for ax in axes[:, 2:].flatten():
    ax.axis('off')


bins = np.linspace(-1, 1, 21)
idx = 0
for area, (fm_idx, fmin, fmax, tmin, tmax) in areas.items():
    # SVM spectrogram coefficients
    ax = axes[idx][0]
    ax.imshow(feature_maps[fm_idx, 1], vmin={0: 0, 1: 0, 2: 0.75}[fm_idx],
              vmax=1, cmap='viridis', aspect='auto')
    fmin_idx = np.argmin(abs(freqs - fmin))
    fmax_idx = max([np.argmin(abs(freqs - fmax)), fmin_idx + 1])
    tmin_idx = np.argmin(abs(times - tmin))
    tmax_idx = np.argmin(abs(times - tmax))
    ax.plot([tmin_idx, tmin_idx, tmax_idx, tmax_idx, tmin_idx],
            [fmin_idx, fmax_idx, fmax_idx, fmin_idx, fmin_idx],
            color='red', linewidth=0.5)
    ax.set_yticks([fmin_idx, fmax_idx])
    ax.set_yticklabels([int(round(freqs[fmin_idx])),
                        f'{int(round(freqs[fmax_idx]))}    '])
    ax.set_ylabel(area, fontsize='small', fontweight='bold')
    ax.invert_yaxis()
    # proportion of area histogram
    ax = axes[idx][1]
    rects = ax.hist(area_contacts[area].values(), bins=bins, color='gray')[2]
    for rect, center in zip(rects, (bins[:-1] + bins[1:]) / 2):
        if center >= prop_thresh:
            rect.set_color('yellow')
        if center <= -prop_thresh:
            rect.set_color('blue')
    ax.set_ylim([0, 50])
    # plot contacts
    brain = mne.viz.Brain(template, **brain_kwargs)
    for name, prop in area_contacts[area].items():
        if abs(prop) > prop_thresh:
            brain._renderer.sphere(
                center=ch_pos['template'][name],
                color='yellow' if prop >= prop_thresh else 'blue',
                scale=0.005)
    for view_idx, view in enumerate(
            (dict(azimuth=60, elevation=100, distance=0.325),
             dict(azimuth=90, elevation=0, distance=0.36),
             dict(azimuth=120, elevation=100, distance=0.325))):
        brain.show_view(**view)
        image = brain.screenshot(mode='rgba')
        image[np.all(image[:, :, :3] == 255, axis=-1), 3] = 0
        axes[idx, view_idx + 2].imshow(image)
    brain.close()
    idx += 1


fig.tight_layout()
fig.subplots_adjust(hspace=0, wspace=0, top=0.97, bottom=0.07, left=0.16)
for axis, name in zip(axes[:, 0], areas):
    axis.set_ylabel(name.replace(' ', '\n'))


# bigger brains
for i, axes2 in enumerate(axes[:, [2, 4]].T):
    for ax in axes2:
        pos = ax.get_position()
        adjust = pos.width * 0.3
        ax.set_position((pos.x0 - adjust / 2, pos.y0 - adjust / 2,
                         pos.width + adjust, pos.height + adjust))


# add view labels, has to be after
for ax_idx, text in enumerate(('Right front', 'Top down', 'Left front')):
    pos = axes[0, ax_idx + 2].get_position()
    fig.text(pos.x0 + pos.width / 2, 0.98, text, ha='center',
             fontsize='large')


axes[0, 1].spines['top'].set_visible(False)
for ax in axes[:, 1]:
    ax.spines['right'].set_visible(False)

for ext in exts:
    fig.savefig(op.join(fig_dir, f'feature_anatomy.{ext}'), dpi=300)

# %%
# Figure 10: Anatomical Locations of Spectral Features

# plot the anatomical locations of each of the time-frequency modulations
# of interest
ignore_keywords = ('unknown', '-vent', 'choroid-plexus', 'vessel',
                   'hypointensities', 'cc_', 'cerebellum')

aseg_img = nib.load(op.join(subjects_dir, template, 'mri', aseg + '.mgz'))
aseg_data = np.array(aseg_img.dataobj)

label_dict = dict()
label_colors = dict()
label_pos = dict()
name_freqs = dict()
raw_labels = set()
# then, go through each area and direction of interest
for name, directions in area_directions.items():
    for direction in directions:
        d_name = u'\u2B06 ' + name if direction == 1 else u'\u2B07 ' + name
        if d_name not in name_freqs:
            name_freqs[d_name] = areas[name][1]
        # finally, go through the area proportions for each electrode and
        # match them up
        for ch_name, prop in area_contacts[name].items():
            if (direction == 1 and prop > prop_thresh) or \
                    (direction == -1 and prop < -prop_thresh):
                these_labels = [label for label in ch_labels[asegs[1]][ch_name]
                                if not any([kw in label.lower() for
                                            kw in ignore_keywords])]
                for label in these_labels:
                    f_label = format_label_dk(label, combine_hemi=True,
                                              cortex=False)
                    if f_label not in label_colors:
                        label_colors[f_label] = colors[label][:3] / 255
                    if f_label not in label_pos:
                        label_pos[f_label] = mne.transforms.apply_trans(
                            aseg_img.header.get_vox2ras_tkr(),
                            np.array(np.where(
                                aseg_data == lut[label])).mean(axis=1))
                    raw_labels.add(label)
                    if d_name in label_dict:
                        label_dict[d_name].add(f_label)
                    else:
                        label_dict[d_name] = set([f_label])


# sort by polar coordinates to wrap frontal to temporal
label_pos_array = np.array(list(label_pos.values()))
# first, rotate axes so left is up so theta can run from -pi to pi
label_pos_rot = mne.transforms.apply_trans(
    mne.transforms.rotation(y=np.pi / 2), label_pos_array)
# then get theta which is really elevation but from -pi to pi
label_pos_theta = mne.transforms._cart_to_sph(label_pos_rot)[:, 1]
# shift from -pi to pi by +pi to 0 to 2 * pi and then shift to the
# phase we want to start with
roi_phase = label_pos_theta[list(label_pos.keys()).index('Putamen')]
label_pos_theta = np.mod((label_pos_theta - roi_phase), 2 * np.pi) - np.pi
# get the order
label_pos_order = dict(zip(label_pos.keys(), label_pos_theta))
labels = sorted(label_pos, key=lambda label: label_pos_order.get(label))
names = sorted(label_dict.keys(), key=lambda name: name_freqs[name])
n_names = len(names)

cmap = plt.get_cmap('Set1')
name_colors = [cmap(i) for i in range(n_names)]
label_cmap = LinearSegmentedColormap.from_list(
    'label_cmap', ['black'] + name_colors, N=n_names + 1)

brain = mne.viz.Brain(template, hemi=None,
                      **dict(brain_kwargs, background='black'))
brain.add_volume_labels(
    aseg, labels=list(raw_labels),
    colors=[label_colors[format_label_dk(label, combine_hemi=True,
                                         cortex=False)]
            for label in raw_labels], fill_hole_size=1)

label_image = np.zeros((len(labels), n_names), dtype=int)
for i, name in enumerate(names):
    for j, label in enumerate(labels):
        if label in label_dict[name]:
            label_image[j, i] = i + 1

fig = plt.figure(figsize=(4, 8), facecolor='black')
gs = fig.add_gridspec(2, 2, height_ratios=(3, 1))

# table of activations
ax = fig.add_subplot(gs[0, :])
ax.imshow(label_image, cmap=label_cmap)
'''
# add second color
d = 0.475  # fudge factor to make triangles look good
for i, name in enumerate(names):
    for j, label in enumerate(labels):
        if label_image[j, i]:
            ax.plot([i - d, i - d, i + d, i + d, i - d],
                    [j - d, j + d, j + d, j - d, j - d],
                    color=label_colors[label], linewidth=0.5)
'''
ax.invert_yaxis()
ax.xaxis.tick_top()
ax.set_xticks(range(n_names))
ax.set_xticklabels(names, color='w', rotation=90)
for i, tick in enumerate(ax.get_xticklabels()):
    tick.set_color(label_cmap(i + 1))
    tick.set_fontsize(8)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, color='w')
for i, tick in enumerate(ax.get_yticklabels()):
    tick.set_color(label_colors[labels[i]])
    if len(labels[i]) > 30:
        tick.set_fontsize(8)

brain.show_view(azimuth=120, elevation=100, distance=0.325)
ax2 = fig.add_subplot(gs[1, 0])
ax2.axis('off')
ax2.imshow(brain.screenshot())
ax2.set_title('Left front', color='w')
brain.show_view(azimuth=80, elevation=180, distance=0.36)
ax3 = fig.add_subplot(gs[1, 1])
ax3.axis('off')
ax3.imshow(brain.screenshot())
ax3.set_title('Bottom up', color='w')

fig.subplots_adjust(hspace=0.15, wspace=0, top=0.75, bottom=0,
                    left=0.05, right=1)
# move table right to make room for labels
pos = ax.get_position()
ax.set_position((pos.x0 + 0.18, pos.y0,
                 pos.width, pos.height))

fig.text(0.02, 0.85, 'a', color='w', fontsize=12)
fig.text(0.02, 0.22, 'b', color='w', fontsize=12)
for ext in exts:
    fig.savefig(op.join(fig_dir, f'feature_table.{ext}'),
                facecolor=fig.get_facecolor(), dpi=300)
