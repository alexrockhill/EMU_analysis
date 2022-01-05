for i in ch_pos.index:
    sub = ch_pos['sub'][i]
    info = mne.io.read_info(op.join(
        subjects_dir, f'sub-{sub}', 'ieeg',
        f'sub-{sub}_template-{template}_task-{task}_info.fif'))
    if ch_pos['elec_name'][i] == 'Event':
        continue
    ch_names = [ch_name.replace(' ', '') for ch_name in info.ch_names]
    ch_idx = ch_names.index(str(ch_pos['elec_name'][i]) + str(int(ch_pos['number'][i])))
    x, y, z = mne.transforms.apply_trans(template_trans, info['chs'][ch_idx]['loc'][:3])
    ch_pos['x'][i], ch_pos['y'][i], ch_pos['z'][i] = x, y, z


ignore_keywords = ('unknown', '-vent', 'choroid-plexus', 'vessel')
fig, axes = plt.subplots(len(areas), 2, figsize=(6, 12), facecolor='black')
hashes = [ax.__hash__() for ax in axes.flatten()]

for (ax, ax2), area in zip(axes, area_contacts):
    pos_labels, neg_labels = dict(), dict()
    for (sub, elec_name, number), prop in area_contacts[area].items():
        pos = ch_pos[(ch_pos['sub'] == sub) &
                     (ch_pos['elec_name'] == elec_name) &
                     (ch_pos['number'] == number)].reset_index().loc[0]
        labels = pos['label'].split(',')
        labels = [label for label in labels if not
                  any(kw in label.lower() for kw in ignore_keywords)]
        if prop > prop_thresh:
            pos_labels[f'Subject {sub} {elec_name}{number}'] = labels
        if prop < -prop_thresh:
            neg_labels[f'Subject {sub} {elec_name}{number}'] = labels
    subplot = hashes.index(ax.__hash__()) + 1
    label_names = set([label for labels in pos_labels.values()
                       for label in labels])
    if label_names:
        mne.viz.plot_channel_labels_circle(
            labels=pos_labels,
            colors={name: colors[name][:3] / 255 for name in label_names},
            fig=fig, subplot=f'{len(areas)}2{subplot}', show=False)
    subplot = hashes.index(ax2.__hash__()) + 1
    label_names = set([label for labels in neg_labels.values()
                       for label in labels])
    if label_names:
        mne.viz.plot_channel_labels_circle(
            labels=neg_labels,
            colors={name: colors[name][:3] / 255 for name in label_names},
            fig=fig, subplot=f'{len(areas)}2{subplot}', show=False)


dark_cmap = plt.get_cmap('Purples')

# proportion of area histogram
ax = axes[idx][1]
rects = ax.hist(area_contacts[area].values(), bins=bins)[2]
for rect, center in zip(rects, (bins[:-1] + bins[1:]) / 2):
    if ((sign == 1 and center >= prop_thresh) or
            (sign == -1 and center <= -prop_thresh)):
        rect.set_color('red')
ax.set_ylim([0, 50])

         dict(azimuth=230, elevation=40, distance=0.2)


name_str = '\n'.join([' '.join([r'$\bf{' + word + '}$'
                                    for word in phrase.split(' ')])
                          for phrase in name.split(',')])

with np.load(op.join(source_dir, 'n_epochs.npz')) as n_epochs:
    n_epochs = {k: v for k, v in n_epochs.items()}


# compute fdr correction
null_images = images['null']
masks = dict()
for sub in subjects:
    null_dist = list()
    for name, null_image in null_images.items():
        sub2, ch = [phrase.split('-')[1] for phrase in
                    name.split('_')[0:2]]
        if sub == int(sub2):
            null_dist.append(abs(null_image))
    null_dist = np.array(null_dist)
    for name, image in images.items():
        sub2, ch = [phrase.split('-')[1] for phrase in
                    name.split('_')[0:2]]
        if sub == int(sub2):
            pvals = np.sum(abs(image) > null_dist, axis=0) / null_dist.shape[0]
            masks[name] = mne.stats.fdr_correction(pvals, alpha=alpha)[0]


np.savez_compressed(op.join(out_dir, 'event_image_masks.npz'),
                    **images['mask'])

        feature_maps[0] += abs(image) > image_threshs[sub]  # count
        feature_maps[1] += image > image_threshs[sub]
        feature_maps[2] += abs(image)
        feature_maps[3] += (abs(image) > image_threshs[sub]) * score


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

ignore_keywords = ('unknown', '-vent', 'choroid-plexus', 'vessel')
best_contact_idx = np.argsort(scores['event_scores'])[-20:][::-1]

fig = plt.figure(figsize=(8, 8), facecolor='black')
labels = {f'Subject {sub}\n{elec_name} {int(number)}':
          [label for label in labels.split(',') if not any(
              kw in label.lower() for kw in ignore_keywords)]
          for sub, elec_name, number, labels in zip(
              ch_pos['sub'][best_contact_idx],
              ch_pos['elec_name'][best_contact_idx],
              ch_pos['number'][best_contact_idx],
              ch_pos['label'][best_contact_idx])}
all_labels = [label for label_list in labels.values()
              for label in label_list]
best_contact_colors = {k: v / 255 for k, v in colors.items()
                       if k in all_labels}
mne.viz.plot_channel_labels_circle(
    labels, best_contact_colors, fig=fig, show=False,
    title='Contacts with the Highest Classification Accuracies')
fig.tight_layout()
fig.savefig(op.join(fig_dir, 'best_contacts.png'), dpi=300)

BANDS = {'evoked': (0, 1), 'delta': (1, 4), 'theta': (4, 8),
         'alpha': (8, 13), 'low_beta': (13, 21),
         'high_beta': (21, 30), 'low_gamma': (30, 60),
         'high_gamma': (60, 250)}

sig_cor = dict()  # signficant correlations by subject
for sub in subjects:
    n_epochs = int(scores[f'sub-{sub}_n_epochs'])
    t = stats.t(n_epochs - 2).interval(1 - alpha)[1]
    x = t**2 / (n_epochs - 2)
    r = np.sqrt(x / (1 - x))
    sig_cor[sub] = r


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
