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
from params import EVENTS as event_dict
from params import LEFT_HANDED_SUBJECTS as lh_sub
from params import FREQUENCIES as freqs
from params import BANDS as bands

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
colors = mne._freesurfer.read_freesurfer_lut()[1]
template_trans = mne.coreg.estimate_head_mri_t(template, subjects_dir)
ch_pos = pd.read_csv(op.join(data_dir, 'derivatives',
                             'elec_contacts_info.tsv'), sep='\t')

# get svm information
source_dir = op.join(data_dir, 'derivatives', 'pca_svm_classifier')
scores = pd.read_csv(op.join(source_dir, 'scores.tsv'), sep='\t')

with np.load(op.join(source_dir, 'n_epochs.npz')) as n_epochs:
    n_epochs = {k: v for k, v in n_epochs.items()}


with np.load(op.join(source_dir, 'event_images.npz')) as images:
    images = {k: v for k, v in images.items()}


with np.load(op.join(source_dir, 'null_images.npz')) as null_images:
    null_images = {k: v for k, v in null_images.items()}


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
fig, axes = plt.subplots(len(subjects), 3, figsize=(4, 12))
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
    info = mne.io.read_info(op.join(
        subjects_dir, f'sub-{sub}', 'ieeg',
        f'sub-{sub}_task-{task}_info.fif'))
    trans = mne.coreg.estimate_head_mri_t(f'sub-{sub}', subjects_dir)
    brain = mne.viz.Brain(f'sub-{sub}', subjects_dir=subjects_dir,
                          cortex='low_contrast', alpha=0.2, background='white')
    brain.add_sensors(info, trans)
    brain.show_view(azimuth=60, elevation=100, distance=300)
    axes[i, 0].imshow(brain.screenshot())
    brain.show_view(azimuth=90, elevation=0)
    axes[i, 1].imshow(brain.screenshot())
    brain.show_view(azimuth=120, elevation=100)
    axes[i, 2].imshow(brain.screenshot())
    brain.close()


fig.subplots_adjust(left=0.07, right=1, top=0.97, bottom=0,
                    wspace=0, hspace=0)
fig.savefig(op.join(fig_dir, 'coverage.png'), dpi=300)


# Figure 3: histogram of classification accuracies with
# binomial null distribution of the number of epochs
# get the number of epochs for each

# EDIT THIS
significant = list()  # alpha = 0.01 significant, uncorrected
significant.append(not stats.binom.cdf(
    n_epochs * score, n_epochs, 0.5) < 1 - alpha)

binsize = 0.01
bins = np.linspace(binsize, 1, int(1 / binsize)) - binsize / 2
fig, ax = plt.subplots()
sig = [score for score, is_sig in
       zip(data_dict['score'], data_dict['sig']) if is_sig]
not_sig = [score for score, is_sig in
           zip(data_dict['score'], data_dict['sig']) if not is_sig]
ax.hist(not_sig, bins=bins, alpha=0.5, color='b',
        density=True, label='not signficant')
ax.hist(sig, bins=bins, alpha=0.5, color='r',
        density=True, label='significant')
dist_mean = np.mean(sig + not_sig)
ax.plot([dist_mean, dist_mean], [0, 25], color='black')
ax.hist(stats.binom.rvs(len(imgs), 0.5, size=10000) / len(imgs), bins=bins,
        color='gray', alpha=0.5, density=True, label='null')
ax.set_xlabel('Test Accuracy')
ax.set_ylabel('Count')
ax.legend()
fig.suptitle('PCA Linear SVM Classification Accuracies')
fig.savefig(op.join(fig_dir, 'score_hist.png'), dpi=300)


# Figure 3: distribution of classification accuracies across
# subjects compared to CSP.

binsize = 0.005
bins = np.linspace(0, 1 - binsize, int(1 / binsize))
for sub in subjects:
    fig, ax = plt.subplots()
    sig = [score for score, sub2, is_sig in
           zip(data_dict['score'], data_dict['sub'],
               data_dict['sig']) if sub == sub2 and is_sig]
    not_sig = [score for score, sub2, is_sig in
               zip(data_dict['score'], data_dict['sub'],
                   data_dict['sig']) if sub == sub2 and not is_sig]
    ax.violinplot(sig + not_sig, [0], vert=False, showextrema=False)
    y = swarm(not_sig, bins=bins) / 50
    ax.scatter(not_sig, y, color='b', label='not signficant')
    y = swarm(sig, bins=bins) / 50
    ax.scatter(sig, y, color='r', label='significant')
    ax.set_xlabel('Test Accuracy')
    ax.set_ylabel('Density')
    ax.legend()
    fig.suptitle('PCA Linear SVM Accuracies')
    fig.subplots_adjust(left=0.18)
    fig.text(0.01, 0.5, f'Subject {sub}', fontsize=14, fontweight='bold',
             rotation='vertical', va='center')
    fig.savefig(op.join(fig_dir, f'sub-{sub}_score_hist.png'), dpi=300)

# Figure 4: Plots of electrodes with high classification accuracies

brain_kwargs = dict(cortex='low_contrast', alpha=0.2, background='white',
                    subjects_dir=subjects_dir, units='m')
brain = mne.viz.Brain(template, **brain_kwargs)

cmap = plt.get_cmap('jet')
for pos, is_sig, score in zip(data_dict['ch_pos'],
                              data_dict['sig'],
                              data_dict['score']):
    if is_sig:
        brain._renderer.sphere(
            center=tuple(pos), color=cmap(score)[:3],
            scale=0.005)

fig, axes = plt.subplots(1, 3, figsize=(8, 4))
for ax in axes.flatten():
    ax.axis('off')
    ax.invert_yaxis()


axes[0].set_title('Right front')
axes[1].set_title('Top down')
axes[2].set_title('Left front')
brain.show_view(azimuth=60, elevation=100, distance=.3)
axes[0].imshow(brain.screenshot())
brain.show_view(azimuth=90, elevation=0)
axes[1].imshow(brain.screenshot())
brain.show_view(azimuth=120, elevation=100)
axes[2].imshow(brain.screenshot())
brain.close()

fig.subplots_adjust(left=0, right=1, top=1, bottom=0,
                    wspace=0, hspace=0)
fig.savefig(op.join(fig_dir, 'high_accuracy.png'), dpi=300)

# save colorbar
fig, ax = plt.subplots(figsize=(1, 6))
gradient = np.linspace(0, 1, 256)
gradient = np.repeat(gradient[:, np.newaxis], 256, axis=1)
ax.imshow(gradient, aspect='auto', cmap=cmap)
ax.set_xticks([])
ax.invert_yaxis()
ax.yaxis.tick_right()
ax.set_yticks(np.array([0, 0.25, 0.5, 0.75, 1]) * 256)
ax.set_yticklabels([0, 0.25, 0.5, 0.75, 1])
fig.tight_layout()
fig.savefig(op.join(fig_dir, 'colorbar.png'))

# Figure 5: Accuracy by label region of interest

ignore_keywords = ('unknown', '-vent', 'choroid-plexus', 'vessel')
labels = np.unique([
    label for labels in data_dict['labels'] for label in labels
    if not any([kw in label.lower() for kw in ignore_keywords])])
label_scores = {label: [score for score, labels in zip(
    data_dict['score'], data_dict['labels']) if label in labels]
    for label in labels}
labels = sorted(labels, key=lambda label: np.mean(label_scores[label]))

fig, ax = plt.subplots(figsize=(8, 12), facecolor='black')
fig.suptitle('Classification Accuracies by Label', color='w')

for idx, label in enumerate(labels):
    rh_scores = [score for score, labels, sub in zip(
        data_dict['score'], data_dict['labels'], data_dict['sub'])
        if sub not in lh_sub and label in labels]
    ax.scatter(rh_scores, [idx] * len(rh_scores),
               color=colors[label])
    lh_scores = [score for score, labels, sub in zip(
        data_dict['score'], data_dict['labels'], data_dict['sub'])
        if sub in lh_sub and label in labels]
    ax.scatter(lh_scores, [idx] * len(lh_scores),
               color=colors[label], marker='^')


ax.set_yticks(range(len(label_scores)))
ax.set_yticklabels(labels)
for tick, label in zip(ax.get_yticklabels(), labels):
    tick.set_color(colors[label])


for tick in ax.get_xticklabels():
    tick.set_color('w')


ax.set_xlabel('Linear SVM Accuracy', color='w')
ax.set_ylabel('Anatomical Label', color='w')

# make legend
ax.text(0.75, 2, 'Subject used right hand', va='center')
ax.scatter([0.95], [2], color='black')
ax.text(0.75, 1, 'Subject used left hand', va='center')
ax.scatter([0.95], [1], marker='^', color='black')
ax.plot([0.74, 0.74, 0.96, 0.96, 0.74], [0.4, 2.6, 2.6, 0.4, 0.4],
        color='black')

fig.tight_layout()
fig.savefig(op.join(fig_dir, 'label_accuracies.png'),
            facecolor=fig.get_facecolor(), dpi=300)

# Figure 7b: Categorization of spectral features

fig, ax = plt.subplots(figsize=(6, 5))
fig.suptitle('Frequency Band Divisions')

image = np.zeros((freqs.size, 2))  # divide time into two
counter = 1
for band in bands:
    fmin, fmax = bands[band]
    for idx in (0, 1):
        image[(freqs >= fmin) & (freqs < fmax), idx] = counter
        counter += 1


ax.imshow(image[::-1], aspect='auto', cmap='tab20',
          vmin=1, vmax=20)
ax.set_xlabel('Time (s)')
ax.set_xticks([-0.5, 0.5, 1.5])
ax.set_xticklabels([-0.5, 0, 0.5])
ax.set_ylabel('Frequency (Hz)')
ax.set_yticks(range(len(freqs)))
ax.set_yticklabels([f'{f}        ' if i % 2 else f for i, f in
                    enumerate(freqs[::-1].round(
                    ).astype(int))], fontsize=8)

fig.tight_layout()
fig.savefig(op.join(fig_dir, 'frequency_division.png'), dpi=300)

# Figure 9: Significant Correlations by Band

sig_cor = dict()  # signficant correlations by subject
for sub in subjects:
    n_epochs = int(scores[f'sub-{sub}_n_epochs'])
    t = stats.t(n_epochs - 2).interval(1 - alpha)[1]
    x = t**2 / (n_epochs - 2)
    r = np.sqrt(x / (1 - x))
    sig_cor[sub] = r


band_cors = list()  # correlations by band
for name, img in imgs.items():
    contact_band_cors = dict(before_pos=list(), before_neg=list(),
                             after_pos=list(), after_neg=list())
    sub, ch = [phrase.split('-')[1] for phrase in
               op.basename(name).split('_')[0:2]]
    sub = int(sub)
    for band_name, (fmin, fmax) in bands.items():
        freq_bool = (freqs >= fmin) & (freqs < fmax)
        if img[freq_bool, :img.shape[1] // 2].max() > sig_cor[sub]:
            contact_band_cors['before_pos'].append(band_name)
        if img[freq_bool, :img.shape[1] // 2].min() < -sig_cor[sub]:
            contact_band_cors['before_neg'].append(band_name)
        if img[freq_bool, img.shape[1] // 2:].max() > sig_cor[sub]:
            contact_band_cors['after_pos'].append(band_name)
        if img[freq_bool, img.shape[1] // 2:].min() < -sig_cor[sub]:
            contact_band_cors['after_neg'].append(band_name)
    band_cors.append(contact_band_cors)

fig, ax = plt.subplots(len(bands), 3)

# Figure 8: Best contacts

best_contacts = sorted(
    {k: v for k, v in scores.items() if 'n_epochs' not in k},
    key=scores.get, reverse=True)[:15]

fig = plt.figure(figsize=(8, 8), facecolor='black')
labels = {contact.replace('sub-', 'Subject ').replace('_ch-', '\nCh '):
          [label for label in anat_labels[contact] if label != 'Unknown']
          for contact in best_contacts}
all_labels = [label for label_list in labels.values()
              for label in label_list]
best_contact_colors = {k: v for k, v in colors.items()
                       if k in all_labels}
mne.viz.plot_channel_labels_circle(
    labels, best_contact_colors, fig=fig,
    title='Contacts with the Highest Classification Accuracies')
fig.savefig(op.join(fig_dir, 'best_contacts.png'), dpi=300)

# Figure 5: best electrode

mean_scores = dict()
for elec_name in electrode_scores:
    mean_scores[elec_name] = np.mean(electrode_scores[elec_name])


best_electrodes = sorted(mean_scores, key=mean_scores.get, reverse=True)[:3]
subs = [elec_name.split('_')[0] for elec_name in best_electrodes]

fig, axes = plt.subplots(1, 3, figsize=(12, 4))

contacts = [contact for contact in anat_labels if
            best_electrodes[0] in contact]
labels = set([label for contact in contacts
              for label in anat_labels[contact]
              if label != 'Unknown' and 'White-Matter' not in label])

brain = mne.viz.Brain(subs[0], **brain_kwargs,
                      title=subs[0].replace('sub-', 'Subject '))
brain.add_volume_labels(aseg=aseg, labels=list(labels))
brain.add_sensors(info, picks=contacts)  # you are here, need info
brain.show_view(azimuth=60, elevation=100, distance=.3)
axes[0].imshow(brain.screenshot())





# TO DO:
# plot rois by accuracy (darker color)
# best electrode labels wheel
# top ten contacts labels plot wheel
# laterality?
