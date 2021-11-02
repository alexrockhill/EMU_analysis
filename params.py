import numpy as np

RAW_DATA_DIR = '/home/alex/SwannLab/EMU_data'
BIDS_ROOT = '/home/alex/SwannLab/EMU_data_BIDS'
DATA_DIR = '/home/alex/SwannLab/EMU_analysis'
SUBJECTS = [1, 2, 5, 6, 9, 10, 11, 12]
TASK = 'SlowFast'
TEMPLATE = 'cvs_avg35_inMNI152'
N_COMPONENTS = 50
EVENTS = dict(baseline=('Fixation', -1.5, -0.5),
              null=('Fixation', -2.499, -1.5),
              event=('Response', -0.5, 0.499),
              go_event=('Go Cue', 0, 1))
ATLAS = 'aparc+aseg'
ALPHA = 0.01
LEFT_HANDED_SUBJECTS = [2, 11]
FREQUENCIES = np.concatenate(
    [np.linspace(1, 10, 10),
     np.logspace(np.log(11), np.log(250), 40, base=np.e)])
BANDS = {'evoked': (0, 1), 'delta': (1, 4), 'theta': (4, 8),
         'alpha': (8, 13), 'low_beta': (13, 21),
         'high_beta': (21, 30), 'low_gamma': (30, 60),
         'high_gamma': (60, 250)}
