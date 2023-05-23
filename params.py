import numpy as np

RAW_DATA_DIR = '../EMU_data'
BIDS_ROOT = '~/mne_data/ds004473'
PLOT_DIR = '.'
EXTENSIONS = ['tiff', 'png']
SUBJECTS = [1, 2, 5, 6, 9, 10, 11, 12]
TASK = 'SlowFast'
TEMPLATE = 'cvs_avg35_inMNI152'
EVENTS = dict(baseline=('Fixation', -1.5, -0.5),
              null=('Fixation', -2.499, -1.5),
              event=('Response', -0.5, 0.499),
              go_event=('Go Cue', 0, 1))
ATLASES = ['aparc.a2009s+aseg',  # Destrieux
           'aparc+aseg']  # Desikan-Killiany
ALPHA = 0.01
LEFT_HANDED_SUBJECTS = [2, 11]
FREQUENCIES = np.concatenate(
    [np.linspace(1, 10, 10),
     np.logspace(np.log(11), np.log(250), 40, base=np.e)])
EXCLUDE_CH = \
    ['sub-1_ch-LENT1', 'sub-1_ch-LENT2', 'sub-1_ch-LENT5',
     'sub-1_ch-LENT6'] + \
    [f'sub-2_ch-{elec}{number}' for elec in ('LAMG', 'LAHP')
     for number in range(1, 5)] + \
    [f'sub-9_ch-LAMY{number}' for number in range(1, 5)] + \
    [f'sub-10_ch-{elec}{number}' for elec in ('LAH', 'LPH')
     for number in range(1, 5)] + \
    [f'sub-11_ch-{elec}{number}' for elec in ('LAH', 'LPH', 'LAMY')
     for number in range(1, 5 if elec == 'LPH' else 6)] + \
    [f'sub-12_ch-{elec}{number}' for elec in ('LPHGB', 'LSTGPH')
     for number in (list(range(1, 6)) + list(range(8, 15)) if
                    elec == 'LPHGB' else range(6, 10))]
