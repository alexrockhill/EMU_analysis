import os
import os.path as op
import numpy as np
import pandas as pd

import mne
import matplotlib.pyplot as plt

from scipy import stats

from params import DATA_DIR as data_dir
from params import BIDS_ROOT as bids_root
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


fig_dir = op.join(data_dir, 'derivatives', 'plots')

if not op.isdir(fig_dir):
    os.makedirs(fig_dir)


# get plotting information
subjects_dir = op.join(bids_root, 'derivatives')
brain_kwargs = dict(cortex='low_contrast', alpha=0.2, background='white',
                    subjects_dir=subjects_dir, units='m')
colors = mne._freesurfer.read_freesurfer_lut()[1]
cmap = plt.get_cmap('viridis')
template_trans = mne.coreg.estimate_head_mri_t(template, subjects_dir)
ch_pos = pd.read_csv(op.join(data_dir, 'derivatives',
                             'elec_contacts_info.tsv'), sep='\t')

# get svm information
source_dir = op.join(data_dir, 'derivatives', 'pca_svm_classifier')
scores = pd.read_csv(op.join(source_dir, 'scores.tsv'), sep='\t')

# remove nans for positions and scores
idx = ~np.logical_or(np.logical_or(np.isnan(
    ch_pos['x']), np.isnan(ch_pos['y'])), np.isnan(ch_pos['z']))
ch_pos = ch_pos[idx].reset_index()
scores = scores[idx].reset_index()

with np.load(op.join(source_dir, 'event_images.npz')) as images:
    images = {k: v for k, v in images.items()}


spec_shape = images[list(images.keys())[0]].shape
times = np.linspace(-0.5, 0.5, spec_shape[1])

with np.load(op.join(source_dir, 'null_images.npz')) as null_images:
    null_images = {k: v for k, v in null_images.items()}


# compute null distribution thresholds
score_threshs = dict()
image_threshs = dict()
for sub in subjects:
    these_scores = scores[scores['sub'] == sub]
    score_threshs[sub] = np.quantile(these_scores['null_scores'], 1 - alpha)
    null_dist = list()
    for name, null_image in null_images.items():
        sub2, ch = [phrase.split('-')[1] for phrase in
                    name.split('_')[0:2]]
        if sub == int(sub2):
            null_dist.append(null_image)
    image_threshs[sub] = np.quantile(
        abs(np.array(null_dist)), 1 - alpha, axis=0)


# feature map computation
feature_maps = np.zeros((4,) + spec_shape)
for sub, elec_name, number, score in zip(
        scores['sub'], scores['elec_name'], scores['number'],
        scores['event_scores']):
    if score > score_threshs[sub]:
        image = images[f'sub-{sub}_ch-{elec_name}{int(number)}']
        feature_maps[0] += abs(image) > image_threshs[sub]  # count
        feature_maps[1] += np.sign(image) * (abs(image) > image_threshs[sub])
        feature_maps[2] += abs(image)
        feature_maps[3] += (abs(image) > image_threshs[sub]) * score


# normalize
feature_maps[1] /= feature_maps[0]  # scale by count
feature_maps[3] /= feature_maps[0]  # scale by count
feature_maps[0] /= feature_maps[0].max()
feature_maps[2] /= feature_maps[2].max()

# time-frequency areas of interest
prop_thresh = 0.5
areas = {'Pre-Movement Beta': (1, 25, 37, -0.4, -0.1),
         'Delta': (1, 1, 5, -0.5, 0.5),
         'Evoked Potential': (1, 0, 0, -0.5, 0.5),
         'High-Beta Rebound': (1, 27, 40, 0, 0.25),
         'Low-Beta Rebound': (1, 14, 23, 0.05, 0.25),
         'Post-Movement Gamma': (1, 45, 160, 0.08, 0.23),
         'Pre-Movement Alpha': (0, 7, 14, -0.3, 0)}


area_contacts = {area: dict() for area in areas}
for name, image in images.items():
    sub, ch = [phrase.split('-')[1] for phrase in
               name.split('_')[0:2]]
    elec_name = ''.join([letter for letter in ch if not letter.isdigit()])
    number = ''.join([letter for letter in ch if letter.isdigit()])
    if not len(ch_pos[(ch_pos['sub'].astype(str) == sub) &
                      (ch_pos['elec_name'] == elec_name) &
                      (ch_pos['number'].astype(int).astype(str) == number)]):
        continue  # no channel position, skip
    mask = (abs(image) > image_threshs[int(sub)]) * np.sign(image)
    for area, (fm_idx, fmin, fmax, tmin, tmax) in areas.items():
        fmin_idx = np.argmin(abs(freqs - fmin))
        fmax_idx = np.argmin(abs(freqs - fmax))
        tmin_idx = np.argmin(abs(times - tmin))
        tmax_idx = np.argmin(abs(times - tmax))
        this_area = mask[slice(fmin_idx, fmax_idx + 1),
                         slice(tmin_idx, tmax_idx + 1)]
        area_contacts[area][(int(sub), elec_name, int(number))] = \
            this_area.sum() / this_area.size


# Plots


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
fig.savefig(op.join(fig_dir, 'task_design.png'), dpi=300)


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


fig.savefig(op.join(fig_dir, 'coverage.png'), dpi=300)


# Figure 3: histogram of classification accuracies with
# binomial null distribution of the number of epochs
# get the number of epochs for each

binsize = 0.01
bins = np.linspace(binsize, 1, int(1 / binsize)) - binsize / 2
fig, ax = plt.subplots()
sig = list()
not_sig = list()
for sub in subjects:
    these_scores = scores[scores['sub'] == sub]
    sig_thresh = np.quantile(these_scores['null_scores'], 1 - alpha)
    sig += [score for score in these_scores['event_scores']
            if score <= sig_thresh]
    not_sig += [score for score in these_scores['event_scores']
                if score > sig_thresh]


ax.hist(sig, bins=bins, alpha=0.5, color='b',
        density=True, label='not signficant')
ax.hist(not_sig, bins=bins, alpha=0.5, color='r',
        density=True, label='significant')
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
fig.savefig(op.join(fig_dir, 'score_hist.png'), dpi=300)

print('Paired t-test p-value: {}'.format(
    stats.ttest_rel(scores['event_scores'], scores['null_scores'])[1]))


# Figure 4: distribution of classification accuracies across
# subjects compared to CSP.

# decoding-specific parameters
csp_freqs = np.logspace(np.log(8), np.log(250), 50, base=np.e)
windows = np.linspace(0, 2, 11)
windows = (windows[1:] + windows[:-1]) / 2  # take mean

fig, axes = plt.subplots(len(subjects) // 2, 4, figsize=(16, 16))
fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.08,
                    hspace=0.2, wspace=0.3)
axes = axes.reshape(len(subjects), 2)
binsize = 0.005
bins = np.linspace(0, 1 - binsize, int(1 / binsize))
for i, sub in enumerate(subjects):
    ax, ax2 = axes[i]
    these_scores = scores[scores['sub'] == sub]
    sig_thresh = np.quantile(these_scores['null_scores'], 1 - alpha)
    sig = [score for score in these_scores['event_scores']
           if score > sig_thresh]
    not_sig = [score for score in these_scores['event_scores']
               if score <= sig_thresh]
    ax.violinplot(sig + not_sig, [0], vert=False, showextrema=False)
    y = swarm(sig, bins=bins) / 50
    ax.scatter(sig, y, color='r', s=2, label='sig')
    y = swarm(not_sig, bins=bins) / 50
    ax.scatter(not_sig, y, color='b', s=2, label='not sig')
    ax.set_ylabel(r'$\bf{Subject' + r'\enspace' + str(sub) + '}$\nDensity')
    ax.axis([0.25, 1, -0.28, 0.28])
    # CSP plot
    tf_scores = np.load(op.join(
        data_dir, 'derivatives', 'csp_decoding',
        f'sub-{sub}_csp_tf_scores.npz'))['arr_0']
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
fig.savefig(op.join(fig_dir, f'svm_csp_comparison.png'), dpi=300)


# Figure 5: Plots of electrodes with high classification accuracies

fig = plt.figure(figsize=(8, 4))
gs = fig.add_gridspec(3, 4)
axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(3)]
                 for i in range(3)])
cax = fig.add_subplot(gs[:, 3])
for ax in axes.flatten():
    ax.axis('off')
    ax.invert_yaxis()


# color contacts by accuracy
brain = mne.viz.Brain(template, **brain_kwargs)

for sub in subjects:
    these_scores = scores[scores['sub'] == sub]
    these_pos = ch_pos[ch_pos['sub'] == sub]
    sig_thresh = np.quantile(these_scores['null_scores'], 1 - alpha)
    for score, x, y, z in zip(these_scores['event_scores'],
                              these_pos['x'], these_pos['y'], these_pos['z']):
        if score > sig_thresh:
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

# plot sampling density by region
ignore_keywords = ('unknown', '-vent', 'choroid-plexus', 'vessel',
                   'white-matter', 'cc_')
densities = dict()
for sub in subjects:
    these_pos = ch_pos[ch_pos['sub'] == sub]
    for these_labels in these_pos['label']:
        if isinstance(these_labels, str):
            for label in these_labels.split(','):
                if any([kw in label.lower() for kw in ignore_keywords]):
                    continue
                if label in densities:
                    densities[label] += 1
                else:
                    densities[label] = 1


label_names = list(densities.keys())
max_count = max(densities.values())
dens_colors = [cmap(densities[name] / max_count) for name in label_names]

brain = mne.viz.Brain(template, **dict(brain_kwargs, alpha=0))
brain.add_volume_labels(aseg=aseg, labels=label_names,
                        colors=dens_colors, alpha=1, smooth=0.9)
brain.show_view(azimuth=60, elevation=100, distance=.3)
axes[1, 0].imshow(brain.screenshot())
brain.show_view(azimuth=90, elevation=0)
axes[1, 1].imshow(brain.screenshot())
brain.show_view(azimuth=120, elevation=100)
axes[1, 2].imshow(brain.screenshot())
brain.close()

# plot accuracy by labels
labels = dict()
for sub in subjects:
    these_scores = scores[scores['sub'] == sub]
    these_pos = ch_pos[ch_pos['sub'] == sub]
    for score, these_labels in zip(these_scores['event_scores'],
                                   these_pos['label']):
        if isinstance(these_labels, str):
            for label in these_labels.split(','):
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
axes[2, 0].imshow(brain.screenshot())
brain.show_view(azimuth=90, elevation=0)
axes[2, 1].imshow(brain.screenshot())
brain.show_view(azimuth=120, elevation=100)
axes[2, 2].imshow(brain.screenshot())
brain.close()

# colorbar
gradient = np.linspace(0.5, 1, 256)
gradient = np.repeat(gradient[:, np.newaxis], 256, axis=1)
cax.imshow(gradient, aspect='auto', cmap=cmap)
cax.set_xticks([])
cax.invert_yaxis()
cax.yaxis.tick_right()
cax.set_yticks(np.array([0, 0.25, 0.5, 0.75, 1]) * 256)
cax.set_yticklabels([0.5, 0.625, 0.75, 0.875, 1])
fig.tight_layout()
pos = cax.get_position()
cax.set_position((pos.x0, 0.15, 0.05, 0.7))

fig.savefig(op.join(fig_dir, 'high_accuracy.png'), dpi=300)


# Figure 6: Accuracy by label region of interest

ignore_keywords = ('unknown', '-vent', 'choroid-plexus', 'vessel')
labels = set([
    label for labels in ch_pos['label'] for label in labels.split(',')
    if not any([kw in label.lower() for kw in ignore_keywords])])
label_scores = {label: [score for score, labels in zip(
    scores['event_scores'], ch_pos['label']) if label in labels.split(',')]
    for label in labels}
labels = sorted(labels, key=lambda label: np.mean(label_scores[label]))

fig, ax = plt.subplots(figsize=(8, 12), facecolor='black')
fig.suptitle('Classification Accuracies by Label', color='w')

for idx, label in enumerate(labels):
    rh_scores = [score for score, labels, sub in zip(
        scores['event_scores'], ch_pos['label'], ch_pos['sub'])
        if sub not in lh_sub and label in labels]
    ax.scatter(rh_scores, [idx] * len(rh_scores),
               color=colors[label][:3] / 255)
    lh_scores = [score for score, labels, sub in zip(
        scores['event_scores'], ch_pos['label'], ch_pos['sub'])
        if sub in lh_sub and label in labels]
    ax.scatter(lh_scores, [idx] * len(lh_scores),
               color=colors[label][:3] / 255, marker='^')


ax.axis([0.25, 1, -0.75, len(labels) - 0.25])
ax.set_yticks(range(len(label_scores)))
ax.set_yticklabels(labels)
for tick, label in zip(ax.get_yticklabels(), labels):
    tick.set_color(colors[label][:3] / 255)
    tick.set_fontsize(8)


for tick in ax.get_xticklabels():
    tick.set_color('w')


ax.set_xlabel('Linear SVM Accuracy', color='w')
ax.set_ylabel('Anatomical Label', color='w')

# make legend
ax.text(0.72, 3, 'Subject used right hand', va='center')
ax.scatter([0.97], [3], color='black')
ax.text(0.72, 1.5, 'Subject used left hand', va='center')
ax.scatter([0.97], [1.5], marker='^', color='black')
ax.plot([0.7, 0.7, 0.99, 0.99, 0.7], [0.4, 4.6, 4.6, 0.4, 0.4],
        color='black')

fig.tight_layout()
fig.subplots_adjust(top=0.95, bottom=0.07)
fig.savefig(op.join(fig_dir, 'label_accuracies.png'),
            facecolor=fig.get_facecolor(), dpi=300)


# Figure 7: Feature maps

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for i, (feature_map, ax) in enumerate(zip(feature_maps, axes)):
    ax.set_xlabel('Time (s)')
    ax.set_xticks(np.linspace(0, spec_shape[1], 5))
    ax.set_xticklabels([-0.5, -0.25, 0, 0.25, 0.5])
    ax.set_yticks(range(len(freqs)))
    ax.set_yticklabels([f'{f}        ' if i % 2 else f for i, f in
                        enumerate(np.array(freqs).round(
                        ).astype(int))], fontsize=8)
    c = ax.imshow(feature_map, vmin={0: 0, 1: -1, 2: 0, 3: 0.5}[i],
                  vmax=1, cmap='viridis', aspect='auto')
    ax.invert_yaxis()
    fig.colorbar(c, ax=ax)


axes[0].set_title('Relative Abundance of\nSignificant Coefficients')
axes[0].set_ylabel('Frequency (Hz)')
fig.text(0.04, 0.95, 'a', fontsize=24)
axes[1].set_title('Proportion of\nSignificant Coefficients')
fig.text(0.52, 0.95, 'b', fontsize=24)
axes[2].set_title('Average Relative Magnitude\nof Coefficients')
fig.text(0.04, 0.47, 'c', fontsize=24)
axes[2].set_ylabel('Frequency (Hz)')
axes[3].set_title('Average Accuracy by\nTime-Frequency')
fig.text(0.52, 0.47, 'd', fontsize=24)

fig.tight_layout()
fig.savefig(op.join(fig_dir, 'feature_map.png'), dpi=300)


# Figure 8: Best contacts

ignore_keywords = ('unknown', '-vent', 'choroid-plexus', 'vessel',
                   'white-matter', 'wm-')
best_contact_idx = np.argsort(scores['event_scores'])[-4:][::-1]

views = [dict(azimuth=250, elevation=60, distance=0.25),
         dict(azimuth=60, elevation=80, distance=0.25),
         dict(azimuth=40, elevation=60, distance=0.3)]

fig, axes = plt.subplots(3, 2, figsize=(6, 8))
axes[-1, 0].set_xlabel('Time (s)')
axes[-1, 0].set_xticks(np.linspace(0, spec_shape[1], 5))
axes[-1, 0].set_xticklabels([-0.5, -0.25, 0, 0.25, 0.5])
for ax in axes[:, 1]:
    ax.axis('off')


for ax in axes[:-1, 0]:
    ax.set_xticks([])


for (ax, ax2), idx, view in zip(axes, best_contact_idx, views):
    sub = ch_pos['sub'][idx]
    elec_name = ch_pos['elec_name'][idx]
    number = ch_pos['number'][idx]
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
    elec_scores = scores[(scores['sub'] == sub) &
                         (scores['elec_name'] == elec_name)]['event_scores']
    locs = np.array(list(montage.get_positions()['ch_pos'].values()))
    # spectrogram plot
    image = images[f'sub-{sub}_ch-{elec_name}{int(number)}']
    mask = abs(image) > image_threshs[sub]
    # mask = binary_opening(binary_closing(mask))  # remove noise
    X, Y = np.meshgrid(range(image.shape[1]), range(image.shape[0]))
    ax.imshow(image, aspect='auto', vmin=-0.05, vmax=0.05,
              cmap='viridis')
    ax.contour(X, Y, mask, levels=[0.5], colors=['r'], alpha=0.25)
    ax.set_yticks(range(len(freqs)))
    ax.set_yticklabels([f'{f}        ' if i % 2 else f for i, f in
                        enumerate(np.array(freqs).round(
                        ).astype(int))], fontsize=6)
    ax.set_ylabel('Frequency (Hz)')
    ax.invert_yaxis()
    # anatomy plot
    brain = mne.viz.Brain(f'sub-{sub}', **dict(brain_kwargs, alpha=0.03))
    for loc, score in zip(locs, elec_scores):
        brain._renderer.sphere(loc, cmap(score * 2 - 1)[:3], 0.005)
    brain.add_volume_labels(aseg='aparc+aseg', labels=labels,
                            alpha=0.5, legend=False, fill_hole_size=1)
    ch_names = [name.replace(' ', '') for name in info.ch_names]  # fix space
    loc = locs[ch_names.index(f'{elec_name}{int(number)}')]
    brain.show_view(focalpoint=loc, **view)
    brain.enable_depth_peeling()
    ax2.imshow(brain.screenshot())
    for label in labels:  # empty plots for legend handling
        ax2.scatter([np.nan], [np.nan], color=colors[label][:3] / 255,
                    marker='s', label=label)
    ax2.legend(loc='lower left', fontsize='xx-small')
    brain.close()


fig.tight_layout()
fig.savefig(op.join(fig_dir, 'best_electrodes.png'), dpi=300)


# Figure 8: Anatomical Locations of Significant Correlations Areas

fig, axes = plt.subplots(len(areas), 5, figsize=(6.5, 10))

axes[-1, 0].set_xticks(np.linspace(0, spec_shape[1], 3))
axes[-1, 0].set_xticklabels([-0.5, 0, 0.5])
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
    ax.imshow(feature_maps[fm_idx], vmin={0: 0, 1: -1, 2: 0, 3: 0.5}[fm_idx],
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
            rect.set_color('blue')
        if center <= -prop_thresh:
            rect.set_color('red')
    ax.set_ylim([0, 50])
    # plot contacts
    brain = mne.viz.Brain(template, **brain_kwargs)
    for (sub, elec_name, number), prop in area_contacts[area].items():
        if prop > prop_thresh or prop < -prop_thresh:
            pos = ch_pos[(ch_pos['sub'] == sub) &
                         (ch_pos['elec_name'] == elec_name) &
                         (ch_pos['number'] == number)].reset_index().loc[0]
            brain._renderer.sphere(
                center=(pos['x'], pos['y'], pos['z']),
                color='blue' if prop > prop_thresh else 'red',
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
axes[0, 0].set_ylabel('Pre-Movement\nBeta')
axes[2, 0].set_ylabel('Evoked\nPotential')
axes[3, 0].set_ylabel('High-Beta\nRebound')
axes[4, 0].set_ylabel('Low-Beta\nRebound')
axes[5, 0].set_ylabel('Post-Movement\nGamma')
axes[6, 0].set_ylabel('Pre-\nMovement\nAlpha')

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


fig.savefig(op.join(fig_dir, 'feature_anatomy.png'), dpi=300)


# Figure 10: Anatomical Locations of Spectral Features

# plot connectivity wheels for beta decrease, high and low beta rebound and
# gamma increase
ignore_keywords = ('unknown', '-vent', 'choroid-plexus', 'vessel',
                   'white-matter', 'wm-', 'cc_', 'cerebellum')

labels = dict()
for sub in subjects:
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
    for ch, these_labels in sub_labels.items():
        ch = ch.replace(' ', '')
        labels[f'sub-{sub}_ch-{ch}'] = \
            [label for label in these_labels if not any(
                [kw in label.lower() for kw in ignore_keywords])]


def format_label(label):
    return label.lower().replace('ctx-', '').replace('lh-', '').replace(
        'rh-', '').replace('left-', '').replace('right-', '').title()


def get_labels(area, direction):
    these_labels = dict()
    for (sub, elec_name, number), prop in area_contacts[area].items():
        ch_labels = labels[f'sub-{sub}_ch-{elec_name}{number}']
        if direction == 1 and prop > prop_thresh:
            these_labels[f'Sub {sub} {elec_name}{number}'] = ch_labels
        if direction == -1 and prop < -prop_thresh:
            these_labels[f'Sub {sub} {elec_name}{number}'] = ch_labels
    label_names = set([label for labels in these_labels.values()
                       for label in labels])
    these_colors = {name: colors[name][:3] / 255 for name in label_names}
    these_labels = {name: [format_label(label) for label in lbls]
                    for name, lbls in these_labels.items()}
    these_colors = {format_label(label): color for label, color in
                    these_colors.items()}
    return these_labels, these_colors


fig, axes = plt.subplots(2, 2, figsize=(8, 8), facecolor='black')
circle_kwargs = dict(fig=fig, show=False, linewidth=1)

these_labels, these_colors = get_labels('Pre-Movement Beta', -1)
mne.viz.plot_channel_labels_circle(
    labels=these_labels, colors=these_colors, subplot='221', **circle_kwargs)
fig.text(0.05, 0.925, 'Pre-Movement\nBeta Decrease', ha='left', color='w')
these_labels, these_colors = get_labels('Low-Beta Rebound', 1)
mne.viz.plot_channel_labels_circle(
    labels=these_labels, colors=these_colors, subplot='222', **circle_kwargs)
fig.text(0.45, 0.9275, 'Low-Beta\nRebound', ha='left', color='w')
these_labels, these_colors = get_labels('High-Beta Rebound', 1)
mne.viz.plot_channel_labels_circle(
    labels=these_labels, colors=these_colors, subplot='223', **circle_kwargs)
fig.text(0.05, 0.45, 'High-Beta\nRebound', ha='left', color='w')
these_labels, these_colors = get_labels('Post-Movement Gamma', 1)
mne.viz.plot_channel_labels_circle(
    labels=these_labels, colors=these_colors, subplot='224', **circle_kwargs)
fig.text(0.45, 0.45, 'Post-Movement\nGamma Increase', ha='left', color='w')

fig.tight_layout()
fig.subplots_adjust(top=0.88)
fig.savefig(op.join(fig_dir, 'feature_labels.png'),
            facecolor=fig.get_facecolor(), dpi=300)
