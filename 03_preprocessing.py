from swann.utils import (get_config, get_layout,
                         exclude_subjects, my_events)
from swann.preprocessing import find_ica, apply_ica, mark_autoreject
from swann.viz import plot_find_bads, plot_ica

config = get_config()
layout = get_layout()

ieegfs = (layout.get(task=config['task'],
                     suffix='ieeg', extension='bdf') +
          layout.get(task=config['task'],
                     suffix='ieeg', extension='edf'))
ieegfs = exclude_subjects(ieegfs)

overwrite_eeg = \
    input('Overwrite preprocessed eeg data if ' +
          'they exist? (y/n)\n').upper() == 'Y'

# loop across subjects
for ieegf in ieegfs:
    plot_find_bads(ieegf, overwrite=overwrite_eeg)

# this will probably take ~5 minutes per subject, probably come back later
for ieegf in ieegfs:
    find_ica(ieegf, overwrite=overwrite_eeg)

# need user input to select out blinks, sacades, heartbeak and muscle artifact
for ieegf in ieegfs:
    plot_ica(ieegf, overwrite=overwrite_eeg)

# this will take even longer ~20+ minutes per subject depending on task length
''' Doesn't work yet, need positions
for ieegf in ieegfs:
    raw = apply_ica(ieegf)
    for event in my_events():
        mark_autoreject(ieegf, event, raw, overwrite=overwrite_eeg)
'''
