# Note, this was run prior to the BIDS formatted data so this will
# not be executable for a reproduction of the analysis as the
# data without events found is not shared
# You will need to run ``pip install pd-parser`` for this to work

import os
import os.path as op

import pd_parser

from params import RAW_DATA_DIR as data_dir
from params import BIDS_ROOT as bids_root
from params import SUBJECTS as subjects
from params import TASK as task

for sub in subjects:
    sub_dir = [op.join(data_dir, d) for d in os.listdir(data_dir) if
               d.endswith('{:03d}'.format(sub))][0]
    edf_fname = [op.join(sub_dir, f) for f in os.listdir(sub_dir)
                 if f.endswith('.edf')][0]
    beh_fname = [op.join(sub_dir, f) for f in os.listdir(sub_dir)
                 if f.endswith('beh.tsv')][0]
    pd_parser.parse_pd(raw=edf_fname, pd_event_name='Fixation',
                       beh=beh_fname, beh_key='fix_onset_time',
                       recover=True)
    pd_parser.add_relative_events(
        raw=edf_fname, beh=beh_fname,
        relative_event_keys=['fix_duration', 'go_time', 'response_time'],
        relative_event_name=['ISI Onset', 'Go Cue', 'Response'])
    pd_parser.save_to_bids(bids_root, edf_fname, str(sub), task=task,
                           beh=beh_fname, data_type='seeg')

# Note: T1 was also manually aligned in freeview to ACPC before sharing
