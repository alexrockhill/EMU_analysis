# This script shares the code that was run to generate the BIDS formatted
# data. It is not be executable for a reproduction of the analysis as the
# data without events found is not shared but may be helpful for converting
# other data.
#
# You will need to run ``pip install pd-parser`` for this to work.

'''

Procedure for importing data/converting to BIDS
1. Convert the .mat data file to a tsv using slowfast_mat2csv.m or
   another task-specific conversion
2. Use ``pd-parser`` to find the events and save to BIDS.
3. Convert MRI and CT from DICOM to nii.gz
    Install from https://people.cas.sc.edu/rorden/mricron/install.html
    and add to path.

    example bash commands:
    cd EMU_data/sub-1_raw
    mkdir MRI_nii
    dcm2niix -o ./MRI_nii -z y ./MRI/
    // you might have to look though them all if you don't know which is good
    freeview MRI_nii/T1.nii.gz
    // add T2/FLAIR for source localization analyses esp if EEG simulatenously
    cp MRI_nii/T1.nii.gz ../../EMU_data_BIDS/sub-1/anat/sub-1_T1w.nii.gz
    cp MRI_nii/T1.json ../../EMU_data_BIDS/sub-1/anat/sub-1_T1w.json

    mkdir CT_nii
    dcm2niix -o ./CT_nii -z y ./CT/
    freeview MRI_nii/CT.nii.gz
    cp MRI_nii/CT.nii.gz ../../EMU_data_BIDS/sub-1/ct/sub-1_ct.nii.gz
    cp MRI_nii/CT.json ../../EMU_data_BIDS/sub-1/ct/sub-1_ct.json
'''

import os
import os.path as op
import numpy as np
from pandas import read_csv
import json

import mne
import mne_bids
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


# compute average response times and accuracties
for sub in subjects:
    df = read_csv(op.join(bids_root, f'sub-{sub}', 'beh',
                          f'sub-{sub}_task-{task}_beh.tsv'), sep='\t')
    if sub == 11:  # subject put hands on wrong keys at from point on
        df = df[:153]
    print('sub-{} Response Time: {} +/- {}'.format(
        sub, np.mean(df['RT']).round(3), np.std(df['RT']).round(3)))
    with open(op.join(bids_root, f'sub-{sub}', 'beh',
                      f'sub-{sub}_task-{task}_beh.json')) as fid:
        meta = json.load(fid)
    correct = df['keypressed'] == np.where(
        df['left_or_right'] == 'right', meta['rightkey'], meta['leftkey'])
    correct = correct[df['keypressed'] != 0]
    print(f'sub-{sub} Accuracy: {(100 * np.mean(correct)).round(1)}%')
    print('sub-{} Missed: {}'.format(sub, (df['keypressed'] == 0).sum()))


# This part was done after 1_gui.py so that channel locations could be used
subjects_dir = op.join(bids_root, 'derivatives')
# Note: T1 was also manually aligned in freeview to ACPC before sharing
for sub in subjects:
    bids_path = mne_bids.BIDSPath(subject=str(sub), task=task, root=bids_root)
    raw = mne_bids.read_raw_bids(bids_path)
    montage = mne.channels.make_dig_montage(
        dict(tmp=[0, 0, 0]), coord_frame='mri')
    montage.dig = mne._freesurfer.get_mni_fiducials(
        f'sub-{sub}', subjects_dir) + montage.dig
    raw.set_montage(montage, on_missing='ignore')
    trans = mne.coreg.estimate_head_mri_t(f'sub-{sub}', subjects_dir)
    t1_fname = op.join(subjects_dir, f'sub-{sub}', 'acpc', 'T1.nii')
    landmarks = mne_bids.get_anat_landmarks(
        t1_fname, info=raw.info, trans=trans, fs_subject=f'sub-{sub}',
        fs_subjects_dir=subjects_dir)
    # save original json data from above (dcm2niix)
    with open(op.join(bids_root, f'sub-{sub}', 'anat',
                      f'sub-{sub}_T1w.json'), 'r') as fid:
        T1_data = json.load(fid)
    mne_bids.write_anat(
        image=t1_fname, bids_path=bids_path.copy().update(suffix='T1w'),
        landmarks=landmarks, deface=dict(inset=15), overwrite=True)
    with open(op.join(bids_root, f'sub-{sub}', 'anat',
                      f'sub-{sub}_T1w.json'), 'r') as fid:
        T1_landmarks = json.load(fid)
    with open(op.join(bids_root, f'sub-{sub}', 'anat',
                      f'sub-{sub}_T1w.json'), 'w') as fid:
        fid.write(json.dumps(dict(**T1_data, **T1_landmarks), indent=4))
