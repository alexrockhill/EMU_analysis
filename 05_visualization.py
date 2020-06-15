import numpy as np

from swann.utils import (get_config, get_layout, get_events,
                         exclude_subjects, get_behf, pick_data,
                         my_events, select_events)
from swann.preprocessing import (apply_ica, get_raw,
                                 mark_autoreject, slowfast2epochs_indices)
from swann.viz import (plot_bursting, plot_power, plot_spectrogram,
                       plot_group_bursting, plot_group_power,
                       plot_burst_shape)

config = get_config()
layout = get_layout()

ieegfs = layout.get(task=config['task'],
                    suffix='ieeg', extension='bdf')
ieegfs += layout.get(task=config['task'],
                     suffix='ieeg', extension='edf')
ieegfs = exclude_subjects(ieegfs)

overwrite = \
    input('Overwrite plots if they exist? (y/n)\n').upper() == 'Y'

np.random.seed(config['seed'])

# loop across subjects
these_events_all = {event: {name: dict() for name in ['All', 'Slow', 'Fast']}
                    for event in my_events()}
for ieegf in ieegfs:
    behf = get_behf(ieegf)
    all_indices, slow_indices, fast_indices = slowfast2epochs_indices(behf)
    raw = get_raw(ieegf)  # raw = apply_ica(ieegf)
    raw.load_data()
    epo_reject_indices = {event: list()  # mark_autoreject(ieegf, event, return_saved=True)
                          for event in my_events()}
    events = get_events(raw, exclude_events=epo_reject_indices)
    picks = [ch for ch in raw.ch_names
             if ('SMA' in ch or 'PM' in ch) and ch not in raw.info['bads']] 
    for event in events:
        these_events = select_events(events[event], all_indices)
        bl_events = select_events(events[config['baseline_event']],
                                  all_indices)
        #for pick in picks:
        #    plot_spectrogram(ieegf, raw, event, these_events, bl_events, ncols=20,
        #                     picks=pick, plot_bursts=True, overwrite=overwrite)
        plot_spectrogram(ieegf, raw, event, these_events, bl_events, method='total',
                         picks=picks, plot_erp=False, overwrite=overwrite)
        plot_spectrogram(ieegf, raw, event, these_events, bl_events, method='phase-locked',
                         picks=picks, plot_erp=False, overwrite=overwrite)
        plot_spectrogram(ieegf, raw, event, these_events, bl_events, method='non-phase-locked',
                         picks=picks, plot_erp=False, overwrite=overwrite)
    '''
    for event in events:
        for name, indices in {'All': all_indices, 'Slow': slow_indices,
                              'Fast': fast_indices}.items():
            these_events = select_events(events[event], indices,
                                         epo_reject_indices[event])
            these_events_all[event][name][ieegf.path] = these_events
        these_events = {name: these_events_all[event][name][ieegf.path]
                        for name in ['All', 'Slow', 'Fast']}
        for pick in picks:
            plot_bursting(ieegf, event, these_events, method='all', picks=[pick], overwrite=overwrite)
    '''
    for event in events:
        for name, indices in {'All': all_indices, 'Slow': slow_indices,
                              'Fast': fast_indices}.items():
            these_events = select_events(events[event], indices,
                                         epo_reject_indices[event])
            these_events_all[event][name][ieegf.path] = these_events
        these_events = {name: these_events_all[event][name][ieegf.path]
                        for name in ['All', 'Slow', 'Fast']}
        for pick in picks:
            plot_power(ieegf, event, these_events, raw.info, picks=[pick], overwrite=overwrite)
        '''plot_burst_shape(eegf, event, {name: these_events},
                         raw.info, overwrite=overwrite)'''

'''
for name, indices in {'All': all_indices, 'Slow': slow_indices,
                      'Fast': fast_indices}.items():
    for event in events:
        plot_group_bursting(eegfs, event, these_events_all[event],
                            overwrite=overwrite)
        plot_group_bursting(eegf, event, these_events_all[event],
                            picks=['C3', 'C4'], overwrite=overwrite)
        plot_group_power(eegfs, event, these_events_all[event],
                         overwrite=overwrite)
        plot_group_power(eegfs, event, these_events_all[event],
                         picks=['C3', 'C4'], overwrite=overwrite)
'''

for event in my_events():
    these_events = {'Slow': these_events_all[event]['Slow'],
                    'Fast': these_events_all[event]['Fast']}
    plot_group_bursting(ieegfs, event, these_events,
                        picks=picks, overwrite=overwrite)
    plot_group_bursting(ieegfs, event, these_events, method='durations',
                        picks=picks, overwrite=overwrite)

from mne import Epochs
from mne.time_frequency import tfr_morlet
event = 'Response'
these_events = select_events(events[event], all_indices,
                             epo_reject_indices[event])
epochs = Epochs(raw, these_events, tmin=config['tmin'], tmax=config['tmax'])
tfr_evo, itc  = tfr_morlet(epochs, np.logspace(np.log10(5), np.log10(250), 20), 7)
for pick in picks:
    fig = tfr_evo.plot(picks=pick, dB=True)
    fig.savefig('sub-1_%s_spectrogram.eps' % pick)
