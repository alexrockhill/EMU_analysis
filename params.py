import numpy as np
import mne

RAW_DATA_DIR = '/home/alex/SwannLab/EMU_data'
BIDS_ROOT = '/home/alex/SwannLab/EMU_data_BIDS'
PLOT_DIR = '/home/alex/SwannLab/EMU_analysis'
EXTENSION = 'tiff'
SUBJECTS = [1, 2, 5, 6, 9, 10, 11, 12]
TASK = 'SlowFast'
TEMPLATE = 'cvs_avg35_inMNI152'
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
