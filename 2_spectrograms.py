import os
import os.path as op
import numpy as np
import matplotlib.pyplot as plt

import mne
import mne_bids
from params import DATA_DIR as data_dir
from params import BIDS_ROOT as bids_root
from params import SUBJECTS as subjects
from params import TASK as task

fig_dir = op.join(data_dir, 'derivatives', 'spectrograms')

if not op.isdir(fig_dir):
    os.makedirs(fig_dir)

for sub in subjects:
    sub = str(sub)
    os.environ['SUBJECT'] = f'sub-{sub}'
    os.environ['SUBJECTS_DIR'] = op.join(bids_root, 'derivatives')
    path = mne_bids.BIDSPath(root=bids_root, subject=sub, task=task)
    raw = mne_bids.read_raw_bids(path)
    raw.pick_types(seeg=True)  # no stim, other channels
    events, event_id = mne.events_from_annotations(raw)
    event_id.pop('ISI Onset')
    # crop first to lessen amount of data, then load
    raw.crop(tmin=events[:, 0].min() / raw.info['sfreq'] - 5,
             tmax=events[:, 0].max() / raw.info['sfreq'] + 5)
    raw.load_data()
    raw.set_eeg_reference('average')

    # plot evoked
    for event, (tmin, tmax) in {'Fixation': (-1, 0), 'Go Cue': (0, 1),
                                'Response': (-0.5, 0.499)}.items():
        epochs = mne.Epochs(raw, events, event_id[event], preload=True,
                            tmin=tmin, tmax=tmax, detrend=1, baseline=None)
        event = event.replace(' ', '')
        fig = epochs.average().plot(show=False)
        fig.savefig(op.join(fig_dir, f'sub-{sub}_event-{event}_evoked.png'))
        plt.close(fig)
        fig = epochs.plot_psd(fmax=250)
        fig.savefig(op.join(fig_dir, f'sub-{sub}_event-{event}_psd.png'))
        plt.close(fig)

    # compute power, do manually for each channel to speed things up
    freqs = np.concatenate(
        [np.linspace(1, 10, 10),
         np.logspace(np.log(11), np.log(250), 40, base=np.e)])
    sfreq = raw.info['sfreq']

    for i, ch in enumerate(raw.ch_names):
        print(str(np.round(100 * i / len(raw.ch_names), 2)) + '% done', ch)
        raw_data = raw._data[i].reshape(1, 1, raw._data.shape[1])
        info = mne.create_info(['0'] + [f'{np.round(f, 2)}' for f in freqs],
                               sfreq, 'seeg')
        raw_tfr = mne.time_frequency.tfr_array_morlet(
            raw_data, sfreq, freqs, output='power', verbose=False)
        raw_tfr = np.concatenate(
            [raw_data[:, :, np.newaxis], raw_tfr], axis=2)  # add DC
        raw_tfr = mne.io.RawArray(raw_tfr.squeeze(), info, raw.first_samp,
                                  verbose=False)
        full_tfr = mne.Epochs(raw_tfr, events, event_id['Fixation'],
                              preload=True, tmin=-1, tmax=2, baseline=None,
                              verbose=False)
        tfr_data = dict()
        for event, (tmin, tmax) in {'Fixation': (-1, 0), 'Go Cue': (0, 1),
                                    'Response': (-0.5, 0.499)}.items():
            tfr = mne.Epochs(raw_tfr, events, event_id[event],
                             preload=True, tmin=tmin, tmax=tmax,
                             baseline=None, verbose=False)
            med = np.median(full_tfr._data, axis=2)[:, :, np.newaxis]
            std = np.std(full_tfr._data, axis=2)[:, :, np.newaxis]
            tfr._data = (tfr._data - med) / std
            tfr_data[event] = tfr._data
            tfr_data[event + ' Times'] = tfr.times
            tfr_evo = mne.time_frequency.AverageTFR(
                raw.info.copy().pick_channels([ch]),
                np.median(tfr._data, axis=0)[np.newaxis],
                tfr.times, [0] + list(freqs), nave=tfr._data.shape[0],
                verbose=False)
            fig = tfr_evo.plot(show=False, verbose=False)[0]
            fig.suptitle(f'{ch} {event} Total Power')
            basename = 'sub-{}_ch-{}_event-{}_spectrogram.png'.format(
                sub, ch.replace(' ', ''), event.replace(' ', ''))
            fig.savefig(op.join(fig_dir, basename))
            plt.close(fig)
        basename = 'sub-{}_ch-{}_spectrogram.npz'.format(
            sub, ch.replace(' ', ''))
        np.savez_compressed(op.join(fig_dir, basename),
                            sfreq=sfreq, freqs=[0] + list(freqs), **tfr_data)
