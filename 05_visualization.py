import numpy as np

from swann.utils import (get_config, get_layout, get_events,
                         exclude_subjects, get_behf,
                         my_events, select_events)
from swann.preprocessing import get_raw, slowfast2epochs_indices
from swann.viz import plot_spectrogram, plot_erp


config = get_config()
layout = get_layout()

ieegfs = layout.get(task=config['task'],
                    suffix='ieeg', extension='vhdr')
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
    # mark_autoreject(ieegf, event, return_saved=True)
    epo_reject_indices = {event: list()
                          for event in my_events()}
    events = get_events(raw, exclude_events=epo_reject_indices)
    picks = [ch for ch in raw.ch_names
             if ('SMA' in ch or 'PM' in ch) and ch not in raw.info['bads']]
    if not picks:
        picks = [raw.ch_names[i] for i in
                 np.random.choice(range(len(raw.ch_names)), 16, replace=False)]
    for event in events:
        these_events = select_events(events[event], all_indices)
        bl_events = select_events(events[config['baseline_event']],
                                  all_indices)
        plot_erp(ieegf, raw, event, these_events, bl_events, picks=picks,
                 overwrite=overwrite)
        '''for pick in picks:
            plot_spectrogram(ieegf, raw, event, these_events, bl_events,
                             ncols=20, picks=pick, plot_bursts=True,
                             overwrite=overwrite)'''
        plot_spectrogram(ieegf, raw, event, these_events, bl_events,
                         method='total', picks=picks, plot_erp=False,
                         overwrite=overwrite)
        plot_spectrogram(ieegf, raw, event, these_events, bl_events,
                         method='phase-locked', picks=picks, plot_erp=False,
                         overwrite=overwrite)
        plot_spectrogram(ieegf, raw, event, these_events, bl_events,
                         method='non-phase-locked', picks=picks,
                         plot_erp=False, overwrite=overwrite)
