import os
import os.path as op
import numpy as np
import pandas as pd

import mne
import nibabel as nib
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.colors import LinearSegmentedColormap

from scipy import stats

from params import PLOT_DIR as plot_dir
from params import BIDS_ROOT as bids_root
from params import EXTENSION as ext
from params import SUBJECTS as subjects
from params import TASK as task
from params import TEMPLATE as template
from params import ATLAS as aseg
from params import ALPHA as alpha
from params import LEFT_HANDED_SUBJECTS as lh_sub
from params import FREQUENCIES as freqs

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
scores = pd.read_csv(op.join(data_dir, 'scores.tsv'), sep='\t')

# load cluster permutation results
with np.load(op.join(data_dir, 'clusters.npz')) as clusters:
    clusters = {k: v for k, v in clusters.items()}


# load SVM images
with np.load(op.join(data_dir, 'event_images.npz')) as images:
    images = {k: v for k, v in images.items()}


spec_shape = images[list(images.keys())[0]].shape
times = np.linspace(-0.5, 0.5, spec_shape[1])

with np.load(op.join(data_dir, 'null_images.npz')) as null_images:
    null_images = {k: v for k, v in null_images.items()}


# compute significant indices pooled across subjects
sig_thresh = np.quantile(scores['null_scores'], 1 - alpha)
not_sig = [i for i, score in enumerate(scores['event_scores'])
           if score <= sig_thresh]
sig = [i for i, score in enumerate(scores['event_scores'])
       if score > sig_thresh]

# compute null distribution thresholds per subject and per image
image_thresh = np.quantile(
    abs(np.array(list(null_images.values()))), 1 - alpha, axis=0)

# feature map computation
feature_maps = np.zeros((3, 2) + spec_shape)
for sub, elec_name, number, score in zip(
        scores['sub'], scores['elec_name'], scores['number'],
        scores['event_scores']):
    name = f'sub-{sub}_ch-{elec_name}{int(number)}'
    image = images[name]
    ch_cluster = clusters[name]
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
prop_thresh = 0.33
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
    sub, ch = [phrase.split('-')[1] for phrase in
               name.split('_')[0:2]]
    elec_name = ''.join([letter for letter in ch if not letter.isdigit()])
    number = ''.join([letter for letter in ch if letter.isdigit()])
    mask = ~np.isnan(cluster) * np.sign(cluster)
    for area, (fm_idx, fmin, fmax, tmin, tmax) in areas.items():
        fmin_idx = np.argmin(abs(freqs - fmin))
        fmax_idx = np.argmin(abs(freqs - fmax))
        tmin_idx = np.argmin(abs(times - tmin))
        tmax_idx = np.argmin(abs(times - tmax))
        this_area = mask[slice(fmin_idx, fmax_idx + 1),
                         slice(tmin_idx, tmax_idx + 1)]
        area_contacts[area][(int(sub), elec_name, int(number))] = \
            np.nansum(this_area) / this_area.size


ch_pos = dict()  # channel positions in template space
template_trans = mne.coreg.estimate_head_mri_t(template, subjects_dir)
for sub in subjects:  # first, find associated labels
    info = mne.io.read_info(op.join(
        subjects_dir, f'sub-{sub}', 'ieeg', f'sub-{sub}_task-{task}_info.fif'))
    montage = mne.channels.make_dig_montage(
        dict(zip(info.ch_names, [ch['loc'][:3] for ch in info['chs']])),
        coord_frame='head')
    montage.apply_trans(template_trans)
    pos = montage.get_positions()['ch_pos']
    for ch_name, this_pos in pos.items():
        ch_name = ch_name.replace(' ', '')
        ch_pos[f'{sub}{ch_name}'] = this_pos


ch_labels = dict()  # channel labels in individual space
for sub in subjects:  # first, find associated labels
    info = mne.io.read_info(op.join(
        subjects_dir, f'sub-{sub}', 'ieeg', f'sub-{sub}_task-{task}_info.fif'))
    trans = mne.coreg.estimate_head_mri_t(f'sub-{sub}', subjects_dir)
    montage = mne.channels.make_dig_montage(
        dict(zip(info.ch_names, [ch['loc'][:3] for ch in info['chs']])),
        coord_frame='head')
    montage.apply_trans(trans)
    sub_labels = mne.get_montage_volume_labels(
        montage, f'sub-{sub}', subjects_dir=subjects_dir,
        aseg=aseg, dist=3)[0]
    for ch_name, labels in sub_labels.items():
        ch_name = ch_name.replace(' ', '')
        ch_labels[f'{sub}{ch_name}'] = labels


def format_label(label, combine_hemi=False, cortex=True):
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


#########
# Plots #
#########

# %%
# Figure 1: Task figure

sr = 800 / 1200  # screen ratio
fig, ax = plt.subplots(figsize=(6, 2))
fig.suptitle('Forced Two-Choice Task Design')
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
ax.text(2.2, -0.55, 'Response Epoch\n-500 to 500 ms',
        va='center', ha='center', fontsize=8, color='red', alpha=0.5)
ax.fill_between([5.1, 6.1], -0.2, 0.2, color='blue', alpha=0.25)
ax.plot([5.13, 5.7, 6.07], [-0.22, -0.38, -0.22], color='blue', alpha=0.25)
ax.text(5.7, -0.55, 'Baseline Epoch\n-1500 to -500 ms',
        va='center', ha='center', fontsize=8, color='blue', alpha=0.5)
ax.fill_between([4.1, 5.1], -0.2, 0.2, color='green', alpha=0.25)
ax.plot([4.13, 4.5, 5.07], [-0.22, -0.68, -0.22], color='green', alpha=0.25)
ax.text(4.5, -0.85, 'Null Epoch\n-2500 to -1500 ms',
        va='center', ha='center', fontsize=8, color='green', alpha=0.5)
fig.savefig(op.join(fig_dir, f'task_design.{ext}'), dpi=300)

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
    brain.add_sensors(info, trans)
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


fig.savefig(op.join(fig_dir, f'coverage.{ext}'), dpi=300)

# %%
# Figure 3: histogram of classification accuracies
#
# Radial basis function scores not shown, almost exactly the same

binsize = 0.01
bins = np.linspace(binsize, 1, int(1 / binsize)) - binsize / 2
fig, ax = plt.subplots()

ax.hist([scores['event_scores'][i] for i in not_sig], bins=bins,
        alpha=0.5, color='b', density=True, label='not signficant')
ax.hist([scores['event_scores'][i] for i in sig], bins=bins,
        alpha=0.5, color='r', density=True, label='significant')
ax.hist(scores['null_scores'], bins=bins, alpha=0.5, color='gray',
        density=True, label='null')
y_bounds = ax.get_ylim()
ax.plot([np.mean(scores['event_scores'])] * 2, y_bounds, color='black')
ax.plot([np.mean(scores['null_scores'])] * 2, y_bounds, color='gray')
ax.set_xlim([0.25, 1])
ax.set_xlabel('Test Accuracy')
ax.set_ylabel('Count')
ax.legend()
fig.suptitle('PCA Linear SVM Classification Accuracies')
fig.savefig(op.join(fig_dir, f'score_hist.{ext}'), dpi=300)

print('Paired t-test p-value: {}'.format(
    stats.ttest_rel(scores['event_scores'], scores['null_scores'])[1]))

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

for score, sub, elec_name, number in zip(scores['event_scores'],
                                         scores['sub'],
                                         scores['elec_name'],
                                         scores['number']):
    if score > sig_thresh:
        x, y, z = ch_pos[f'{sub}{elec_name}{number}']
        brain._renderer.sphere(center=(x, y, z),
                               color=cmap(score * 2 - 1)[:3],
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
                   'white-matter', 'wm-', 'cc_', 'cerebellum')

labels = dict()
for score, sub, elec_name, number in zip(scores['event_scores'],
                                         scores['sub'],
                                         scores['elec_name'],
                                         scores['number']):
    these_labels = ch_labels[f'{sub}{elec_name}{number}']
    for label in these_labels:
        if any([kw in label.lower() for kw in ignore_keywords]):
            continue
        if label in labels:
            labels[label].append(score)
        else:
            labels[label] = [score]


label_names = list(labels.keys())
acc_colors = [cmap(np.mean(labels[name]) * 2 - 1) for name in label_names]

brain = mne.viz.Brain(template, **dict(brain_kwargs, alpha=0))
brain.add_volume_labels(aseg=aseg, labels=label_names,
                        colors=acc_colors, alpha=1, smooth=0.9)
brain.show_view(azimuth=60, elevation=100, distance=.3)
axes[1, 0].imshow(brain.screenshot())
brain.show_view(azimuth=90, elevation=0)
axes[1, 1].imshow(brain.screenshot())
brain.show_view(azimuth=120, elevation=100)
axes[1, 2].imshow(brain.screenshot())
brain.close()
fig.text(0.1, 0.55, 'b')

# colorbar
gradient = np.linspace(0.5, 1, 256)
gradient = np.repeat(gradient[:, np.newaxis], 256, axis=1)
cax.imshow(gradient, aspect='auto', cmap=cmap)
cax.set_xticks([])
cax.invert_yaxis()
cax.yaxis.tick_right()
cax.set_yticks(np.array([0, 0.25, 0.5, 0.75, 1]) * 256)
cax.set_yticklabels([0.5, 0.625, 0.75, 0.875, 1])
cax.yaxis.set_label_position('right')
cax.set_ylabel('Accuracy')

# plot counts of electrodes per area
counts = dict()
for these_labels in ch_labels.values():
    for label in these_labels:
        if any([kw in label.lower() for kw in ignore_keywords]):
            continue
        if label in counts:
            counts[label] += 1
        else:
            counts[label] = 1


density_colors = [cmap(min([counts[name] / 10, 1.])) for name in label_names]

brain = mne.viz.Brain(template, **dict(brain_kwargs, alpha=0))
brain.add_volume_labels(aseg=aseg, labels=label_names,
                        colors=density_colors, alpha=1, smooth=0.9)
brain.show_view(azimuth=60, elevation=100, distance=.3)
axes[2, 0].imshow(brain.screenshot())
brain.show_view(azimuth=90, elevation=0)
axes[2, 1].imshow(brain.screenshot())
brain.show_view(azimuth=120, elevation=100)
axes[2, 2].imshow(brain.screenshot())
brain.close()
fig.text(0.1, 0.3, 'c')

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
fig.savefig(op.join(fig_dir, f'high_accuracy.{ext}'), dpi=300)

# %%
# Figure 5: Accuracy by label region of interest

ignore_keywords = ('unknown', '-vent', 'choroid-plexus', 'vessel', 'cc_',
                   'wm', 'cerebellum')  # signal won't cross dura
labels = set([label for labels in ch_labels.values() for label in labels
              if not any([kw in label.lower() for kw in ignore_keywords])])
label_scores = dict()
for score, sub, elec_name, number in zip(scores['event_scores'],
                                         scores['sub'],
                                         scores['elec_name'],
                                         scores['number']):
    these_labels = ch_labels[f'{sub}{elec_name}{number}']
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
        for name, idxs in {'sig': sig, 'not_sig': not_sig}.items():
            these_scores = [score for i, (sub, elec_name, number, score) in
                            enumerate(zip(scores['sub'], scores['elec_name'],
                                          scores['number'],
                                          scores['event_scores']))
                            if label in ch_labels[f'{sub}{elec_name}{number}']
                            and i in idxs and (lh == (sub in lh_sub))]
            color = colors[label][:3] / 255
            if color.mean() > 0.9:
                color *= 0.75  # gray out white
            # triangle if left hand used, hollow if not significant
            ax.scatter(these_scores, [idx] * len(these_scores),
                       color=color, marker='^' if lh else None,
                       facecolors=None if name == 'sig' else 'none')


ax.axis([0.25, 1, -0.75, len(labels) - 0.25])
ax.set_yticks(range(len(label_scores)))
ax.set_yticklabels([format_label(label) for label in labels])
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


ax.set_xlabel('Linear SVM Accuracy', color='w')
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
    these_scores = scores[scores['sub'] == sub]
    these_sig = [score for score in these_scores['event_scores']
                 if score > sig_thresh]
    these_not_sig = [score for score in these_scores['event_scores']
                     if score <= sig_thresh]
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


fig.suptitle('CSP-SVM Comparison by Subject')
fig.savefig(op.join(fig_dir, f'svm_csp_comparison.{ext}'), dpi=300)

# %%
# Figure 7: Best contacts

ignore_keywords = ('unknown', '-vent', 'choroid-plexus', 'vessel',
                   'white-matter', 'wm-')
best_contact_idx = np.argsort(scores['event_scores'])[-3:][::-1]

views = [dict(azimuth=35, elevation=70, distance=0.25),
         dict(azimuth=60, elevation=80, distance=0.25),
         dict(azimuth=40, elevation=60, distance=0.3)]

fig, axes = plt.subplots(3, 2, figsize=(6, 8))
axes[-1, 0].set_xlabel('Time (s)')
for ax in axes[:, 1]:
    ax.axis('off')


for ax in axes[:, 0]:
    ax.set_xticks(np.linspace(0, spec_shape[1] - 1, 5))
    ax.set_xticklabels([-0.5, -0.25, 0, 0.25, 0.5])


for (ax, ax2), idx, view in zip(axes, best_contact_idx, views):
    sub = scores['sub'][idx]
    elec_name = scores['elec_name'][idx]
    number = scores['number'][idx]
    score = scores['event_scores'][idx]
    ax.set_title(f'Subject {sub} {elec_name} {int(number)} '
                 'Test Accuracy {:.2f}'.format(score))
    info = mne.io.read_info(op.join(
        subjects_dir, f'sub-{sub}', 'ieeg', f'sub-{sub}_task-{task}_info.fif'))
    info.pick_channels([ch for ch in info.ch_names if elec_name in ch])
    trans = mne.coreg.estimate_head_mri_t(f'sub-{sub}', subjects_dir)
    montage = mne.channels.make_dig_montage(
        dict(zip(info.ch_names, [ch['loc'][:3] for ch in info['chs']])),
        coord_frame='head')
    montage.apply_trans(trans)
    labels = mne.get_montage_volume_labels(
        montage, f'sub-{sub}', subjects_dir=subjects_dir,
        aseg=aseg, dist=5)[0].values()
    labels = set([label for these_labels in labels for label in these_labels
                  if not any([kw in label.lower() for kw in ignore_keywords])])
    locs = np.array(list(montage.get_positions()['ch_pos'].values()))
    # spectrogram plot
    image = images[f'sub-{sub}_ch-{elec_name}{int(number)}']
    cluster = clusters[f'sub-{sub}_ch-{elec_name}{int(number)}']
    mask = ~np.isnan(cluster)
    X, Y = np.meshgrid(range(image.shape[1]), range(image.shape[0]))
    img = ax.imshow(image, aspect='auto', vmin=-0.05, vmax=0.05,
                    cmap='viridis')
    ax.contour(X, Y, mask, levels=[0.5], colors=['r'], alpha=0.25)
    ax.set_yticks(range(len(freqs)))
    ax.set_yticklabels([f'{f}        ' if i % 2 else f for i, f in
                        enumerate(np.array(freqs).round(
                        ).astype(int))], fontsize=6)
    ax.set_ylabel('Frequency (Hz)')
    ax.invert_yaxis()
    fig.colorbar(img, ax=ax)
    # anatomy plot
    brain = mne.viz.Brain(f'sub-{sub}', **dict(brain_kwargs, alpha=0.25))
    for loc, name in zip(locs, montage.ch_names):
        is_best = int(name.replace(elec_name, '').replace(' ', '')) == number
        brain._renderer.sphere(loc, 'black' if is_best else 'gray', 0.005)
    brain.add_volume_labels(aseg='aparc+aseg', labels=labels,
                            alpha=0.5, legend=False, fill_hole_size=1)
    ch_names = [name.replace(' ', '') for name in info.ch_names]  # fix space
    loc = locs[ch_names.index(f'{elec_name}{int(number)}')]
    brain.show_view(focalpoint=loc, **view)
    brain.enable_depth_peeling()
    ax2.imshow(brain.screenshot())
    for label in labels:  # empty plots for legend handling
        ax2.scatter([np.nan], [np.nan], color=colors[label][:3] / 255,
                    marker='s', label=format_label(label))
    ax2.legend(loc='lower left', fontsize='xx-small')
    brain.close()


fig.tight_layout()
fig.savefig(op.join(fig_dir, f'best_electrodes.{ext}'), dpi=300)

# %%
# Figure 8: Feature maps

fig, axes = plt.subplots(3, 2, figsize=(8, 8))
for idx, ((svm_map, cluster_map), (ax1, ax2)) in enumerate(
        zip(feature_maps, axes)):
    vmin = 0 if idx < 2 else 0.5
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
        ax1.set_title('Relative Abundance of\nSignificant Coefficients')
        ax2.set_title('Relative Abundance of\nSignificant Clusters')
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
    ax.imshow(feature_maps[fm_idx, 1], vmin={0: 0, 1: 0, 2: 0, 3: 0.5}[fm_idx],
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
    for (sub, elec_name, number), prop in area_contacts[area].items():
        if prop > prop_thresh or prop < -prop_thresh:
            brain._renderer.sphere(
                center=ch_pos[f'{sub}{elec_name}{number}'],
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
        for (sub, elec_name, number), prop in area_contacts[name].items():
            if (direction == 1 and prop > prop_thresh) or \
                    (direction == -1 and prop < -prop_thresh):
                these_labels = [label for label in
                                ch_labels[f'{sub}{elec_name}{number}']
                                if not any([kw in label.lower() for
                                            kw in ignore_keywords])]
                for label in these_labels:
                    f_label = format_label(label, combine_hemi=True,
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
roi_phase = label_pos_theta[list(label_pos.keys()).index('Putamen')] + np.pi
label_pos_theta = np.mod((label_pos_theta + np.pi - roi_phase), 2 * np.pi)
# get the order
label_pos_order = dict(zip(label_pos.keys(), label_pos_theta))
labels = sorted(label_pos, key=lambda label: label_pos_order.get(label))
names = sorted(label_dict.keys(), key=lambda name: name_freqs[name])
n_names = len(names)

cmap = plt.get_cmap('Set1')
name_colors = [cmap(i) for i in range(n_names)]
label_cmap = LinearSegmentedColormap.from_list(
    'label_cmap', name_colors, N=n_names)

brain = mne.viz.Brain(template, hemi=None,
                      **dict(brain_kwargs, background='black'))
brain.add_volume_labels(
    aseg, labels=list(raw_labels),
    colors=[label_colors[format_label(label, combine_hemi=True,
                                      cortex=False)]
            for label in raw_labels], fill_hole_size=1)


fig, axes = plt.subplots(5, 3, figsize=(8, 12), facecolor='black',
                         subplot_kw=dict(projection='polar'))
gs = axes[0, 0].get_gridspec()  # for adjustments later

node_angles = mne.viz.circular_layout(
    ['pattern'] + labels, ['pattern'] + labels,
    start_pos=90 - (360 / (len(labels) + 3)),
    group_boundaries=[0, 1])

for ax, name in zip(axes.flatten(), names):
    node_names = [name] + labels
    con = np.zeros((len(node_names), len(node_names))) * np.nan
    for label in label_dict[name]:
        node_idx = node_names.index(label)
        label_color = names.index(name) / n_names
        con[0, node_idx] = con[node_idx, 0] = label_color  # symmetric

    node_colors = [name_colors[names.index(name)]] + \
        [label_colors[label] for label in labels]

    mne.viz.circle._plot_connectivity_circle(
        con, [''] * len(node_names), node_angles=node_angles, title=name,
        node_colors=node_colors, node_height=4,
        vmin=0, vmax=1, colormap=label_cmap,
        textcolor=name_colors[names.index(name)], colorbar=False, linewidth=1,
        ax=ax, show=False)

brain.show_view(azimuth=120, elevation=100, distance=0.325)
axes[3, 2].remove()  # switch these two out to cartesian
axes[3, 2] = fig.add_subplot(gs[3, 2])
axes[3, 2].imshow(brain.screenshot())
brain.show_view(azimuth=80, elevation=180, distance=0.36)
axes[4, 2].remove()
axes[4, 2] = fig.add_subplot(gs[4, 2])
axes[4, 2].imshow(brain.screenshot())

axes[3, 2].set_title('Left front', color='w')
axes[4, 2].set_title('Bottom up', color='w')

# add plot to bottom left 4 plots
for ax in axes[3:, :2].flatten():
    ax.remove()  # remove small axes
ax = fig.add_subplot(gs[3:, :2], polar=True)  # add back a big axis
pos = ax.get_position()
mne.viz.circle._plot_connectivity_circle(
    np.zeros(con.shape) * np.nan, [''] + labels, node_angles=node_angles,
    node_colors=node_colors, node_height=4, vmin=0, vmax=1, fontsize_names=8,
    colormap=label_cmap, textcolor='white', colorbar=False, linewidth=1,
    ax=ax, show=False)

fig.subplots_adjust(hspace=0.1, wspace=0, top=0.95, bottom=0, left=0, right=1)
# adjust big axis, bring in
ax.set_position((pos.x0 + 0.05, pos.y0 + 0.02,
                 pos.width - 0.1, pos.height - 0.1))

fig.text(0.02, 0.98, 'a', color='w', fontsize=12)
fig.text(0.02, 0.38, 'b', color='w', fontsize=12)
fig.savefig(op.join(fig_dir, f'feature_labels.{ext}'),
            facecolor=fig.get_facecolor(), dpi=300)
