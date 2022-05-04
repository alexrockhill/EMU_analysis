import os.path as op
import numpy as np
import matplotlib.pyplot as plt

import mne
import mne_bids


def load_raw(bids_root, sub, task, subjects_dir):
    path = mne_bids.BIDSPath(root=bids_root, subject=str(sub), task=task)
    raw = mne_bids.read_raw_bids(path)
    info = mne.io.read_info(op.join(
        subjects_dir, f'sub-{sub}', 'ieeg',
        f'sub-{sub}_task-{task}_info.fif'))
    raw.drop_channels([ch for ch in raw.ch_names if ch not in info.ch_names])
    raw.info = info
    events, event_id = mne.events_from_annotations(raw)
    raw.pick_types(seeg=True)  # no stim, other channels
    # crop first to lessen amount of data, then load
    raw.crop(tmin=events[:, 0].min() / raw.info['sfreq'] - 5,
             tmax=events[:, 0].max() / raw.info['sfreq'] + 5)
    raw.load_data()
    raw.set_eeg_reference('average')
    return raw


def plot_evoked(raw, sub, event_dict, plot_dir):
    # plot evoked
    events, event_id = mne.events_from_annotations(raw)
    for name, (event, tmin, tmax) in event_dict.items():
        epochs = mne.Epochs(raw, events, event_id[event], preload=True,
                            tmin=tmin, tmax=tmax, detrend=1, baseline=None)
        fig = epochs.average().plot(show=False)
        fig.savefig(op.join(plot_dir, f'sub-{sub}_event-{name}_evoked.png'))
        plt.close(fig)
        fig = epochs.plot_psd(fmax=250, show=False)
        fig.savefig(op.join(plot_dir, f'sub-{sub}_event-{name}_psd.png'))
        plt.close(fig)


def compute_tfr(raw, i, raw_filtered, freqs, event_dict):
    sfreq = raw.info['sfreq']
    ch = raw.ch_names[i]
    events, event_id = mne.events_from_annotations(raw)
    raw_data = raw.get_data(picks=[ch]).reshape(1, 1, raw._data.shape[1])
    raw_tfr = mne.time_frequency.tfr_array_morlet(
        raw_data, sfreq, freqs, output='power', verbose=False)
    raw_tfr = np.concatenate(
        [raw_filtered._data[i].reshape(1, 1, 1, raw._data.shape[1]),
         raw_tfr], axis=2)  # add DC
    info = mne.create_info(['0'] + [f'{np.round(f, 2)}' for f in freqs],
                           sfreq, 'seeg')
    raw_tfr = mne.io.RawArray(raw_tfr.squeeze(), info, raw.first_samp,
                              verbose=False)
    # use time from beginning of the first event to end of the last event
    full_tfr = mne.Epochs(raw_tfr, events, event_id['Fixation'],
                          preload=True, tmin=-2.5, tmax=2, baseline=None,
                          verbose=False)
    tfr_data = dict()
    for name, (event, tmin, tmax) in event_dict.items():
        tfr = mne.Epochs(raw_tfr, events, event_id[event],
                         preload=True, tmin=tmin, tmax=tmax,
                         baseline=None, verbose=False)
        med = np.median(full_tfr._data, axis=2)[:, :, np.newaxis]
        std = np.std(full_tfr._data, axis=2)[:, :, np.newaxis]
        tfr._data = (tfr._data - med) / std
        tfr_data[name] = dict(data=tfr._data, times=tfr.times,
                              sfreq=sfreq, freqs=[0] + list(freqs))
    return tfr_data


def plot_tfr(tfr_data, info, ch, sub, event, plot_dir):
    for name in tfr_data:
        tfr_evo = mne.time_frequency.AverageTFR(
            info.copy().pick_channels([ch]),
            np.median(tfr_data[name]['data'], axis=0)[np.newaxis],
            tfr_data[name]['times'], tfr_data[name]['freqs'],
            nave=tfr_data[name]['data'].shape[0],
            verbose=False)
        fig = tfr_evo.plot(show=False, verbose=False)[0]
        fig.suptitle(f'{ch} {event} Total Power')
        basename = 'sub-{}_ch-{}_event-{}'.format(
            sub, ch.replace(' ', ''), name)
        fig.savefig(op.join(plot_dir, basename + '_spectrogram.png'))
        plt.close(fig)


def plot_image(fig, ax, img, freqs, times, vmin=None, vmax=None,
               cmap='RdYlBu_r', cbar=True):  # helper function
    vmin = img.min() if vmin is None else vmin
    vmax = img.max() if vmax is None else vmax
    c = ax.imshow(img, vmin=vmin, vmax=vmax, cmap=cmap, aspect='auto')
    ax.invert_yaxis()
    ax.set_xlabel('Time (s)')
    ticks = np.linspace(0, times.size - 1, 5).round().astype(int)
    ax.set_xticks(np.linspace(0, times.size, 5))
    ax.set_xticklabels(times[ticks].round(2))
    ax.set_ylabel('Frequency (Hz)')
    ax.set_yticks(range(len(freqs)))
    ax.set_yticklabels([f'{f}        ' if i % 2 else f for i, f in
                        enumerate(np.array(freqs).round(
                        ).astype(int))], fontsize=8)
    if cbar:
        fig.colorbar(c)
    fig.tight_layout()
    return c


def plot_confusion_matrix(out_fname, tfr_data, tp, fp, tn, fn):
    # coefficients plot
    fig, ax = plt.subplots()
    fig.suptitle(f'Subject {sub} {ch} {event_dict[bl_event][0]}-'
                 f'{event_dict[event][0]} Linear SVM'
                 '\nClassification Feature Importances '
                 f'({score.round(2)})')
    plot_image(fig, ax, image, tfr_data[event]['freqs'],
               tfr_data[event]['times'], vmin=-0.05, vmax=0.05)
    fig.savefig(op.join(plot_dir, out_fname + '_features.png'))
    plt.close(fig)
    # spectrograms by classification plot
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(f'{ch} {event_dict[event][0]} Spectrograms Based '
                 f'on Classification Accuracy ({score.round(2)})')
    plot_image(fig, axes[0, 0],
               np.median(tfr_data[event]['data'][tp], axis=0),
               tfr_data[event]['freqs'], tfr_data[event]['times'],
               vmin=-1, vmax=1, cbar=False, cmap='RdYlBu_r')
    axes[0, 0].set_title(
        f'True {event_dict[event][0].title()} ({len(tp)})')
    plot_image(fig, axes[0, 1],
               np.median(tfr_data[event]['data'][fp], axis=0),
               tfr_data[event]['freqs'], tfr_data[event]['times'],
               vmin=-1, vmax=1, cbar=False, cmap='RdYlBu_r')
    axes[0, 0].set_xlabel('')
    axes[0, 1].set_title(
        f'False {event_dict[event][0].title()} ({len(fp)})')
    plot_image(fig, axes[1, 1],
               np.median(tfr_data[bl_event]['data'][tn], axis=0),
               tfr_data[bl_event]['freqs'],
               tfr_data[bl_event]['times'],
               vmin=-1, vmax=1, cbar=False, cmap='RdYlBu_r')
    axes[0, 1].set_xlabel('')
    axes[1, 1].set_title(
        f'True {event_dict[bl_event][0].title()} ({len(tn)})')
    c = plot_image(fig, axes[1, 0],
                   np.median(tfr_data[bl_event]['data'][fn], axis=0),
                   tfr_data[bl_event]['freqs'],
                   tfr_data[bl_event]['times'],
                   vmin=-1, vmax=1, cbar=False, cmap='RdYlBu_r')
    axes[1, 0].set_title(
        f'False {event_dict[bl_event][0].title()} ({len(fn)})')
    fig.subplots_adjust(top=0.9, right=0.9, bottom=0.07, hspace=0.2)
    cax = fig.add_axes([0.915, 0.1, 0.01, 0.8])
    fig.colorbar(c, cax=cax)
    cax.set_ylabel(r'Power ($\mu$$V^{2}$)', labelpad=0)
    fig.savefig(op.join(plot_dir, out_fname + '_comparison.png'))
    plt.close(fig)
    # single random example spectrograms plot
    n_show = min([tp.size, fp.size, tn.size, fn.size])
    n_col = int(n_show ** 0.5)
    for name, idx in {f'True {event}': tp, f'False {event}': fp,
                      f'True {bl_event}': tn,
                      f'False {bl_event}': fn}.items():
        fig, axes = plt.subplots(n_col, n_show // n_col + 1,
                                 figsize=(15, 15))
        fig.suptitle(f'{name} ({len(idx)})')
        for i, ax in zip(np.random.choice(idx, n_show, replace=False),
                         axes.flatten()):
            if y_test[i]:
                plot_image(fig, ax, tfr_data[event]['data'][i],
                           tfr_data[event]['freqs'],
                           tfr_data[event]['times'],
                           vmin=-5, vmax=5, cmap='RdYlBu_r',
                           cbar=False)
            else:
                plot_image(fig, ax, tfr_data[bl_event]['data'][i],
                           tfr_data[bl_event]['freqs'],
                           tfr_data[bl_event]['times'],
                           vmin=-5, vmax=5, cmap='RdYlBu_r',
                           cbar=False)
        for ax in axes.flatten():
            ax.axis('off')
        name_str = name.replace(' ', '_').lower()
        fig.tight_layout()
        fig.savefig(
            op.join(plot_dir, out_fname + f'_{name_str}.png'),
            dpi=300)
        plt.close(fig)
