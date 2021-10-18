import os
import os.path as op
import numpy as np
import json

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
from params import EVENT as event

source_dir = op.join(data_dir, 'derivatives',
                     f'pca_{event.lower()}_classifier')
fig_dir = op.join(data_dir, 'derivatives', 'plots')

subjects_dir = op.join(bids_root, 'derivatives')

if not op.isdir(fig_dir):
    os.makedirs(fig_dir)


with np.load(op.join(source_dir, 'scores.npz')) as scores:
    scores = {k: v for k, v in scores.items()}


with np.load(op.join(source_dir, 'imgs.npz')) as imgs:
    imgs = {k: v for k, v in imgs.items()}


trans = mne.coreg.estimate_head_mri_t(template, subjects_dir)
# get plotting information
ch_positions = dict()  # positions in template space
anat_labels = dict()  # labels in individual space
anat_scores = dict()  # scores per label
colors = dict()  # the color for each label
electrode_scores = dict()  # scores per electrode
for sub in subjects:
    info = mne.io.read_info(op.join(
        subjects_dir, f'sub-{sub}', 'ieeg',
        f'sub-{sub}_template-{template}_task-{task}_info.fif'))
    montage = mne.channels.make_dig_montage(
        dict(zip(info.ch_names, [ch['loc'][:3] for ch in info['chs']])),
        coord_frame='head')
    montage.apply_trans(trans)
    labels, colors2 = mne.get_montage_volume_labels(
        montage, f'sub-{sub}', subjects_dir=subjects_dir,
        aseg=aseg)
    colors.update(colors2)
    ch_pos = montage.get_positions()['ch_pos']
    for ch_data in imgs:  # bit of a hack to match subject IDs
        sub2, ch = [phrase.split('-')[1] for phrase in
                    op.basename(ch_data).split('_')[0:2]]
        if int(sub2) == sub:
            ch2 = {ch2.replace(' ', ''): ch2 for ch2 in info.ch_names}[ch]
            score = float(scores[f'sub-{sub}_ch-{ch}'])
            ch_positions[f'sub-{sub}_ch-{ch}'] = ch_pos[ch2]
            anat_labels[f'sub-{sub}_ch-{ch}'] = labels[ch2]
            for label in labels[ch2]:
                if label in anat_scores:
                    anat_scores[label].append(score)
                else:
                    anat_scores[label] = [score]
            elec_name = f'sub-{sub}_ch-' + ''.join(
                [letter for letter in ch2 if
                 not letter.isdigit()]).rstrip()
            if elec_name in electrode_scores:
                electrode_scores[elec_name].append(score)
            else:
                electrode_scores[elec_name] = [score]


# Figure 1: Individual implant plots to show sampling
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


# Figure 2: histogram of classification accuracies with
# binomial null distribution of the number of epochs
# get the number of epochs for each
dist = dict(sig=list(), not_sig=list())
for ch_data in imgs:
    sub, ch = [phrase.split('-')[1] for phrase in
               op.basename(ch_data).split('_')[0:2]]
    n_epochs = int(scores[f'sub-{sub}_n_epochs'])
    score = float(scores[f'sub-{sub}_ch-{ch}'])
    if stats.binom.cdf(n_epochs * score, n_epochs, 0.5) < 1 - alpha:
        dist['not_sig'].append(score)
    else:
        dist['sig'].append(score)


binsize = 0.01
bins = np.linspace(binsize, 1, int(1 / binsize)) - binsize / 2
fig, ax = plt.subplots()
ax.hist(dist['not_sig'], bins=bins, alpha=0.5, color='r',
        density=True, label='not signficant')
ax.hist(dist['sig'], bins=bins, alpha=0.5, color='b',
        density=True, label='significant')
dist_mean = np.mean(dist['sig'] + dist['not_sig'])
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
dist = {str(sub): dict(sig=list(), not_sig=list()) for sub in subjects}
for ch_data in imgs:
    sub, ch = [phrase.split('-')[1] for phrase in
               op.basename(ch_data).split('_')[0:2]]
    n_epochs = int(scores[f'sub-{sub}_n_epochs'])
    score = float(scores[f'sub-{sub}_ch-{ch}'])
    if stats.binom.cdf(n_epochs * score, n_epochs, 0.5) < 1 - alpha:
        dist[sub]['not_sig'].append(score)
    else:
        dist[sub]['sig'].append(score)


def swarm(x, bins):
    counts = np.ones((bins.size))
    y = np.zeros((len(x)))
    for i, this_x in enumerate(x):
        idx = np.where(this_x < bins)[0][0] - 1
        y[i] = counts[idx] // 2 if counts[idx] % 2 else -counts[idx] // 2
        counts[idx] += 1
    return y


binsize = 0.005
bins = np.linspace(0, 1 - binsize, int(1 / binsize))
for sub in dist:
    fig, ax = plt.subplots()
    ax.violinplot(dist[sub]['sig'] + dist[sub]['not_sig'], [0],
                  vert=False, showextrema=False)
    y = swarm(dist[sub]['not_sig'], bins=bins) / 50
    ax.scatter(dist[sub]['not_sig'], y, color='b', label='not signficant')
    y = swarm(dist[sub]['sig'], bins=bins) / 50
    ax.scatter(dist[sub]['sig'], y, color='r', label='significant')
    ax.set_xlabel('Test Accuracy')
    ax.set_ylabel('Density')
    ax.legend()
    fig.suptitle('PCA Linear SVM Accuracies')
    fig.subplots_adjust(left=0.18)
    fig.text(0.01, 0.5, f'Subject {sub}', fontsize=14, fontweight='bold',
             rotation='vertical', va='center')
    fig.savefig(op.join(fig_dir, f'sub-{sub}_score_hist.png'), dpi=300)

# Figure 4: Plots of electrodes with high classification accuracies
# based on their time-frequency characteristics.

#   Part 1: all electrodes over 0.75 classification, colored by score.

# plot electrodes with high accuracies
brain_kwargs = dict(cortex='low_contrast', alpha=0.2, background='white',
                    subjects_dir=subjects_dir, units='m')
brain = mne.viz.Brain(template, **brain_kwargs)

cmap = plt.get_cmap('jet')
for ch_data in imgs:
    sub, ch = [phrase.split('-')[1] for phrase in
               op.basename(ch_data).split('_')[0:2]]
    score = scores[f'sub-{sub}_ch-{ch}']
    if score > 0.75:
        x, y, z = ch_positions[f'sub-{sub}_ch-{ch}']
        brain._renderer.sphere(
            center=(x, y, z), color=cmap(score)[:3],
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

# Figure 4: Best contacts

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


# Figure X: white matter vs gray matter

wm = np.concatenate([anat_scores[label] for label in anat_scores
                     if 'White-Matter' in label])
gm = np.concatenate([
    anat_scores[label] for label in anat_dict if
    all([kw not in label for kw in ('White-Matter', 'WM', 'Unknown')])])
p = stats.ttest_ind(wm, gm)[1]
fig, ax = plt.subplots()
fig.suptitle('White Matter-Grey Matter Classifications, p={:.3f}'.format(p))
vdict = ax.violinplot([wm, gm], [0, 1], showextrema=False)
x = swarm(wm, bins=bins) / 50
ax.scatter(x, wm, color='b', s=1)
vdict['bodies'][0].set_facecolor('b')
x = swarm(gm, bins=bins) / 50
ax.scatter(1 + x, gm, color='r', s=1)
vdict['bodies'][1].set_facecolor('r')
ax.set_xticks([0, 1])
ax.set_xticklabels(['White Matter', 'Grey Matter'])
ax.set_ylabel('Linear SVM Accuracy')
fig.savefig(op.join(fig_dir, 'wm_vs_gm.png'), dpi=300)


# TO DO:
# plot rois by accuracy (darker color)
# best electrode labels wheel
# top ten contacts labels plot wheel
# laterality?
