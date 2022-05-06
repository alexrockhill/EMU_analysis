import sys
import os
import os.path as op
import matplotlib.pyplot as plt
import numpy as np

from utils import get_subjects, load_raw, bipolar_reference

from params import BIDS_ROOT as bids_root
from params import PLOT_DIR as plot_dir
from params import SUBJECTS as subjects
from params import TASK as task

subjects_dir = op.join(bids_root, 'derivatives')

psd_dir = op.join(plot_dir, 'derivatives', 'psd_plots')
if not op.isdir(psd_dir):
    os.makedirs(psd_dir)

# can run one subject at a time to save memory/parallelize
for sub in get_subjects(subjects, __name__, sys.argv):
    raw = load_raw(bids_root, sub, task, subjects_dir)
    fig = raw.plot_psd(fmax=300, show=False)
    fig.savefig(op.join(psd_dir, f'sub-{sub}_task-{task}_psd_av_ref.png'))
    plt.close(fig)
    raw = bipolar_reference(raw)
    fig = raw.plot_psd(fmax=300, show=False)
    fig.savefig(op.join(psd_dir, f'sub-{sub}_task-{task}_psd_bipolar.png'))
    plt.close(fig)
    raw = load_raw(bids_root, sub, task, subjects_dir)
    ref = raw.ch_names[np.argmin(np.var(raw._data, axis=1))]
    raw.set_eeg_reference([ref])
    raw.drop_channels([ref])
    fig = raw.plot_psd(fmax=300, show=False)
    fig.savefig(op.join(psd_dir, f'sub-{sub}_task-{task}_psd_quiet_ref.png'))
    plt.close(fig)
