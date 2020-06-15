from swann.utils import (get_config, get_layout,
                         exclude_subjects)
from swann.preprocessing import apply_ica
from swann.analyses import decompose_tfr, find_bursts

config = get_config()
layout = get_layout()

ieegfs = (layout.get(task=config['task'],
                     suffix='ieeg', extension='bdf') +
          layout.get(task=config['task'],
                     suffix='ieeg', extension='vhdr'))
ieegfs = exclude_subjects(ieegfs)

overwrite = \
    input('Overwrite analysis data if ' +
          'they exist? (y/n)\n').upper() == 'Y'

# loop across subjects: may take ~10 minutes each
'''
for ieegf in ieegfs:
    raw = apply_ica(ieegf)
    decompose_tfr(ieegf, raw, overwrite=overwrite)  # defaults to beta
'''

for ieegf in ieegfs:
    tfr, ch_names, sfreq = decompose_tfr(ieegf, return_saved=True)
    find_bursts(ieegf, tfr,  ch_names, thresh=3, overwrite=overwrite)
