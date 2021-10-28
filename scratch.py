for sub in [5, 9, 10]:
    path.update(subject=str(sub))
    raw = mne_bids.read_raw_bids(path)
    raw.set_montage(None)
    CT_aligned = nib.load(op.join(
        subjects_dir, f'sub-{sub}', 'CT', 'CT_aligned.mgz'))
    info = mne.io.read_info(op.join(
        subjects_dir, f'sub-{sub}', 'ieeg',
        f'sub-{sub}_task-{task}_info.fif'))
    trans = mne.coreg.estimate_head_mri_t(f'sub-{sub}', subjects_dir)
    raw.info = info
    gui = mne.gui.locate_ieeg(raw.info, trans, CT_aligned,
                              subject=f'sub-{sub}', subjects_dir=subjects_dir)
    while input('Finished, save to disk? (y/N)\t') != 'y':
        mne.io.write_info(op.join(subjects_dir, f'sub-{sub}', 'ieeg',
                                  f'sub-{sub}_task-{task}_info.fif'),
                          raw.info)


binsize = 0.005
bins = np.linspace(0, 1 - binsize, int(1 / binsize))

wm_keywords = ('white-matter', 'wm', 'cc_')
ignore_keywords = ('unknown', '-vent', 'choroid-plexus', 'vessel')
wm_labels = [label for label in anat_scores
             if any([kw in label.lower() for kw in wm_keywords])]
wm = [score for label in wm_labels for score in anat_scores[label]]
gm_labels = [label for label in anat_scores
             if not any([kw in label.lower() for kw in wm_keywords]) and
             not any([kw in label.lower() for kw in ignore_keywords])]
gm = [score for label in gm_labels for score in anat_scores[label]]
p = stats.ttest_ind(wm, gm)[1]

all_scores = {label: tuple(anat_scores[label]) for label in
              wm_labels + gm_labels}

fig, ax = plt.subplots(figsize=(8, 12), facecolor='black')
fig.suptitle('Classification Accuracies by Label', color='w')
all_labels = sorted(
    all_scores, key=lambda label: np.mean(all_scores[label]))
for idx, label in enumerate(all_labels):
    ax.scatter(all_scores[label], [idx] * len(all_scores[label]),
               color=colors[label])


ax.set_yticks(range(len(all_scores)))
ax.set_yticklabels(all_labels)
for tick, label in zip(ax.get_yticklabels(), all_labels):
    tick.set_color(colors[label])


for tick in ax.get_xticklabels():
    tick.set_color('w')


ax.set_xlabel('Linear SVM Accuracy', color='w')
ax.set_ylabel('Anatomical Label', color='w')
fig.tight_layout()
fig.savefig(op.join(fig_dir, 'label_accuracies.png'),
            facecolor=fig.get_facecolor(), dpi=300)

fig, ax = plt.subplots()
fig.suptitle('White Matter-Grey Matter Classifications, p={:.3f}'.format(p))
vdict = ax.violinplot([wm, gm], [0, 1], showextrema=False)
x = swarm(wm, bins=bins) / 50
ax.scatter(x, wm, color='b', s=1)
ax.plot([-0.4, 0.4], [np.mean(wm), np.mean(wm)], color='b')
vdict['bodies'][0].set_facecolor('b')
x = swarm(gm, bins=bins) / 50
ax.scatter(1 + x, gm, color='r', s=1)
ax.plot([0.6, 1.4], [np.mean(gm), np.mean(gm)], color='r')
vdict['bodies'][1].set_facecolor('r')
ax.set_xticks([0, 1])
ax.set_xticklabels([f'White Matter (N={len(wm)})',
                    f'Grey Matter (N={len(gm)})'])
ax.set_ylabel('Linear SVM Accuracy')
fig.savefig(op.join(fig_dir, 'wm_vs_gm.png'), dpi=300)

# Figure 3: Plots of electrodes with high classification accuracies
# based on their time-frequency characteristics.

#   Part 1: all electrodes over 0.75 classification, colored by score.

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
