import os
import os.path as op
import numpy as np
import json

import mne
import matplotlib.pyplot as plt

from scipy import stats

alpha = 0.01
event = 'Response'
data_dir = f'./derivatives/pca_{event.lower()}_classifier'
fig_dir = f'./derivatives/plots'
atlas = 'desikan-killiany'
template = 'fsaverage'
subs = [1, 2, 5, 6, 9, 10, 11, 12]
subjects_dir = '/home/alex/SwannLab/EMU_data_BIDS/derivatives'
os.environ['SUBJECTS_DIR'] = subjects_dir

if not op.isdir(fig_dir):
    os.makedirs(fig_dir)


with np.load(op.join(data_dir, 'scores.npz')) as scores:
    scores = {k: v for k, v in scores.items()}


with np.load(op.join(data_dir, 'imgs.npz')) as imgs:
    imgs = {k: v for k, v in imgs.items()}


trans = mne.coreg.estimate_head_mri_t('fsaverage', subjects_dir)
# get plotting information
elec_pos = dict()
elec_labels = dict()
anat_dict = dict()
for sub in subs:
    os.environ['SUBJECT'] = f'sub-{sub}'
    raw = mne.io.read_raw_fif(op.join(
        subjects_dir, f'sub-{sub}', 'ieeg',
        f'sub-{sub}_template-{template}_task-SlowFast_ieeg.fif'),
        preload=False)
    montage = raw.get_montage()
    montage.apply_trans(trans)
    ch_pos = montage.get_position()['ch_pos']
    for ch_data in imgs:
        sub2, ch = [phrase.split('-')[1] for phrase in
                    op.basename(ch_data).split('_')[0:2]]
        if int(sub2) == sub:
            ch2 = {ch2.replace(' ', ''): ch2 for ch2 in ch_pos}[ch]
            label = json.loads(elecs[ch2][4])[atlas]
            score = float(scores[f'sub-{sub}_ch-{ch}'])
            elec_pos[f'sub-{sub}_ch-{ch}'] = elecs[ch2][:3]
            elec_labels[f'sub-{sub}_ch-{ch}'] = label
            if label in anat_dict:
                anat_dict[label].append(score)
            else:
                anat_dict[label] = [score]


# Figure 1: histogram of classification accuracies with
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


# Figure 2: distribution of classification accuracies across
# subjects compared to CSP.
dist = {str(sub): dict(sig=list(), not_sig=list()) for sub in subs}
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


# Figure 3: Plots of electrodes with high classification accuracies
# based on their time-frequency characteristics.

#   Part 1: all electrodes over 0.75 classification, colored by score.

with np.load('./derivatives/elec_pos.npz') as elec_pos:
    elec_pos = {k: v for k, v in elec_pos.items()}


def plot_brain(rois):
    renderer = mne.viz.backends.renderer.create_3d_figure(
        size=(1200, 900), bgcolor='w', scene=False)
    mne.viz.set_3d_view(figure=renderer.figure, distance=500,
                        azimuth=None, elevation=None)
    for roi in rois:
        renderer.mesh(*roi.vert.T, triangles=roi.tri, color=roi.color,
                      opacity=roi.opacity, representation=roi.representation)
    return renderer


# plot electrodes with high accuracies
rois = get_rois('all', template=template, opacity=0.1)
renderer = plot_brain(rois)

cmap = plt.get_cmap('jet')
for ch_data in imgs:
    sub, ch = [phrase.split('-')[1] for phrase in
               op.basename(ch_data).split('_')[0:2]]
    score = scores[f'sub-{sub}_ch-{ch}']
    if score > 0.75:
        x, y, z = elec_pos[f'sub-{sub}_ch-{ch}']
        renderer.sphere(center=(x, y, z), color=cmap(score)[:3],
                        scale=5)


# renderer.screenshot(op.join(fig_dir, 'high_accuracy.png'))

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

#   Part 2: all electrodes with pre-movement high-beta decreases.

renderer = plot_brain(rois)
freqs = np.concatenate(
    [[0], np.linspace(1, 10, 10),
     np.logspace(np.log(11), np.log(250), 40, base=np.e)])
times = np.linspace(-0.5, 4.999, 1000)
for ch_data in imgs:
    sub, ch = [phrase.split('-')[1] for phrase in
               op.basename(ch_data).split('_')[0:2]]
    if scores[f'sub-{sub}_ch-{ch}'] > 0.75:
        beta = imgs[ch_data][np.logical_and(freqs > 13, freqs < 40)]
        beta = beta[:, np.logical_and(times > -0.25, times < 0)]
        if beta.min() < -0.025:
            print(f'sub-{sub}_ch-{ch}', elec_labels[f'sub-{sub}_ch-{ch}'])
            x, y, z = elec_pos[f'sub-{sub}_ch-{ch}']
            renderer.sphere(center=(x, y, z), color='b', scale=5)


# renderer.screenshot(op.join(fig_dir, 'beta_decrease.png'))


#   Part 3: all electrodes with post-movement gamma increases.

renderer = plot_brain(rois)
# ax.plot([250, 500, 500, 250, 250], [13, 13, 40, 40, 13])
for ch_data in imgs:
    sub, ch = [phrase.split('-')[1] for phrase in
               op.basename(ch_data).split('_')[0:2]]
    if scores[f'sub-{sub}_ch-{ch}'] > 0.75:
        gamma = imgs[ch_data][np.logical_and(freqs > 50, freqs < 120)]
        gamma = gamma[:, np.logical_and(times > 0, times < 0.25)]
        if gamma.max() > 0.025:
            print(f'sub-{sub}_ch-{ch}', elec_labels[f'sub-{sub}_ch-{ch}'])
            x, y, z = elec_pos[f'sub-{sub}_ch-{ch}']
            renderer.sphere(center=(x, y, z), color='r', scale=5)


# renderer.screenshot(op.join(fig_dir, 'gamma_increase.png'))


#   Part 4: all electrodes with pre-movement delta increases.

renderer = plot_brain(rois)
for ch_data in imgs:
    sub, ch = [phrase.split('-')[1] for phrase in
               op.basename(ch_data).split('_')[0:2]]
    if scores[f'sub-{sub}_ch-{ch}'] > 0.75:
        delta = imgs[ch_data][np.logical_and(freqs > 1, freqs < 4)]
        delta = delta[:, times < 0]
        if delta.max() > 0.025:
            print(f'sub-{sub}_ch-{ch}', elec_labels[f'sub-{sub}_ch-{ch}'])
            x, y, z = elec_pos[f'sub-{sub}_ch-{ch}']
            renderer.sphere(center=(x, y, z), color='r', scale=5)


# renderer.screenshot(op.join(fig_dir, 'delta_increase.png'))

# Figure 4: Make a wordcloud with areas that have better
# scores with exponentially greater representation,
# and compare white matter to gray matter.

wm = np.concatenate([anat_dict[label] for label in anat_dict
                     if 'White-Matter' in label])
gm = np.concatenate([
    anat_dict[label] for label in anat_dict if
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


N = 1000
text = ''
for label in anat_dict:
    score = anat_dict[label].max()
    if score > 0.75:
        n = np.round(N * np.exp(score - 1)).astype(int)
        label = label.replace('ctx-', '')
        text += ' '.join([label] * n)
wordcloud = WordCloud().generate(text)
fig, ax = plt.subplots()
ax.imshow(wordcloud, interpolation='bilinear')
ax.axis('off')
fig.savefig(op.join(fig_dir, 'wordcloud.png'), dpi=300)