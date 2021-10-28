import os
import os.path as op
import numpy as np

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


template_trans = mne.coreg.estimate_head_mri_t(template, subjects_dir)
# get plotting information
subject = list()
electrode_name = list()  # name of the electrode shaft
contact_number = list()  # number of contact
ch_position = list()  # positions in template space
anat_labels = list()  # labels in individual space
contact_score = list()  # scores per electrode
significant = list()  # alpha = 0.01 significant, uncorrected

colors = dict()  # stores colors from all subjects
for sub in subjects:
    # get labels from individual subject anatomy
    info = mne.io.read_info(op.join(
        subjects_dir, f'sub-{sub}', 'ieeg',
        f'sub-{sub}_task-{task}_info.fif'))
    montage = mne.channels.make_dig_montage(
        dict(zip(info.ch_names, [ch['loc'][:3] for ch in info['chs']])),
        coord_frame='head')
    trans = mne.coreg.estimate_head_mri_t(f'sub-{sub}', subjects_dir)
    montage.apply_trans(trans)
    labels, colors2 = mne.get_montage_volume_labels(
        montage, f'sub-{sub}', subjects_dir=subjects_dir,
        aseg=aseg, dist=1)  # use colors here
    colors.update(colors2)
    # get positions from template-warped anatomy
    info = mne.io.read_info(op.join(
        subjects_dir, f'sub-{sub}', 'ieeg',
        f'sub-{sub}_template-{template}_task-{task}_info.fif'))
    montage = mne.channels.make_dig_montage(
        dict(zip(info.ch_names, [ch['loc'][:3] for ch in info['chs']])),
        coord_frame='head')
    montage.apply_trans(template_trans)
    ch_pos = montage.get_positions()['ch_pos']
    for ch_data in imgs:  # bit of a hack to match subject IDs
        sub2, ch = [phrase.split('-')[1] for phrase in
                    op.basename(ch_data).split('_')[0:2]]
        if int(sub2) == sub:
            subject.append(sub)
            ch2 = {ch2.replace(' ', ''): ch2 for ch2 in info.ch_names}[ch]
            electrode_name.append(''.join([letter for letter in ch2 if
                                           not letter.isdigit()]).rstrip())
            contact_number.append(''.join([letter for letter in ch2 if
                                           letter.isdigit()]).rstrip())
            ch_position.append(ch_pos[ch2])
            anat_labels.append(labels[ch2])
            score = float(scores[f'sub-{sub}_ch-{ch}'])
            contact_score.append(score)
            n_epochs = int(scores[f'sub-{sub}_n_epochs'])
            significant.append(not stats.binom.cdf(
                n_epochs * score, n_epochs, 0.5) < 1 - alpha)


data_dict = dict(sub=subject, elec_name=electrode_name,
                 number=contact_number, ch_pos=ch_position,
                 labels=anat_labels, score=contact_score,
                 sig=significant)  # holds all data


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
