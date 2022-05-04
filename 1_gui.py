import os
import os.path as op
import numpy as np
import nibabel as nib

import mne
import mne_bids

from params import BIDS_ROOT as bids_root
from params import SUBJECTS as subjects
from params import TASK as task
from params import TEMPLATE as template

subjects_dir = op.join(bids_root, 'derivatives')
path = mne_bids.BIDSPath(root=bids_root, task=task)
out_dir = op.join(bids_root, 'derivatives', 'analysis_data')

# %%
# Align CT, takes ~15 minutes per subject, no user input
for sub in subjects:
    T1 = nib.load(op.join(subjects_dir, f'sub-{sub}', 'mri', 'T1.mgz'))
    CT_orig = nib.load(op.join(bids_root, f'sub-{sub}', 'anat',
                               f'sub-{sub}_ct.nii.gz'))
    reg_affine, _ = mne.transforms.compute_volume_registration(
        CT_orig, T1, pipeline='rigids')
    np.savez_compressed(op.join(
        subjects_dir, f'sub-{sub}', 'CT', 'reg_affine.npz'),
        reg_affine=reg_affine)
    CT_aligned = mne.transforms.apply_volume_registration(
        CT_orig, T1, reg_affine)
    nib.save(CT_aligned, op.join(
        subjects_dir, f'sub-{sub}', 'CT', 'CT_aligned.mgz'))


# %%
# For a few subjects the above didn't work and so this was used to
# align (11 and 12). This has since been fixed in MNE-Python (using
# manual pre-alignment)
'''
import ants
T1 = ants.image_read(op.join(subjects_dir, f'sub-{sub}', 'mri', 'T1.mgz'))
CT_orig = ants.image_read(op.join(bids_root, f'sub-{sub}', 'anat',
                                  f'sub-{sub}_ct.nii.gz'))
trans = ants.registration(fixed=T1, moving=CT_orig, type_of_transform='Rigid')
ants.image_write(trans['warpedmovout'],
                 op.join(subjects_dir, f'sub-{sub}',
                         'CT', 'CT_aligned_test.mgz'))
CT_orig = nib.load(op.join(bids_root, f'sub-{sub}', 'anat',
                           f'sub-{sub}_ct.nii.gz'))
CT_aligned = nib.load(op.join(subjects_dir, f'sub-{sub}',
                              'CT', 'CT_aligned.mgz'))
reg_affine, _ = mne.transforms.compute_volume_registration(
    CT_orig, CT_aligned, pipeline='rigids')
np.savez_compressed(op.join(
    subjects_dir, f'sub-{sub}', 'CT', 'reg_affine.npz'),
    reg_affine=reg_affine)
'''

# %%
# Pick contact locations, requires user input
for sub in subjects:
    path.update(subject=str(sub))
    raw = mne_bids.read_raw_bids(path)
    raw.set_montage(None)
    T1 = nib.load(op.join(subjects_dir, f'sub-{sub}', 'mri', 'T1.mgz'))
    CT_aligned = nib.load(op.join(
        subjects_dir, f'sub-{sub}', 'CT', 'CT_aligned.mgz'))
    trans = mne.coreg.estimate_head_mri_t(f'sub-{sub}', subjects_dir)
    gui = mne.gui.locate_ieeg(raw.info, trans, CT_aligned,
                              subject=f'sub-{sub}', subjects_dir=subjects_dir)
    while input('Finished, save to disk? (y/N)\t') != 'y':
        mne.io.write_info(op.join(subjects_dir, f'sub-{sub}', 'ieeg',
                                  f'sub-{sub}_task-{task}_info.fif'),
                          raw.info)

# %%
# Warp to template, takes ~15 minutes per subject, no user input
template_subjects_dir = op.join(os.environ['FREESURFER_HOME'], 'subjects')
if not op.exists(op.join(subjects_dir, template)):
    os.symlink(op.join(template_subjects_dir, template),
               op.join(subjects_dir, template))


template_brain = nib.load(
    op.join(subjects_dir, template, 'mri', 'brain.mgz'))
template_trans = mne.coreg.estimate_head_mri_t(template, subjects_dir)
for sub in subjects:
    path.update(subject=str(sub))
    raw = mne_bids.read_raw_bids(path)
    info = mne.io.read_info(op.join(
        subjects_dir, f'sub-{sub}', 'ieeg', f'sub-{sub}_task-{task}_info.fif'))
    raw.drop_channels([ch for ch in raw.ch_names if ch not in info.ch_names])
    raw.info = info
    trans = mne.coreg.estimate_head_mri_t(f'sub-{sub}', subjects_dir)
    subject_brain = nib.load(
        op.join(subjects_dir, f'sub-{sub}', 'mri', 'brain.mgz'))
    reg_affine, sdr_morph = mne.transforms.compute_volume_registration(
        subject_brain, template_brain, verbose=True)
    montage = raw.get_montage()
    montage.apply_trans(trans)
    CT_aligned = nib.load(op.join(
        subjects_dir, f'sub-{sub}', 'CT', 'CT_aligned.mgz'))
    montage_warped, elec_image, warped_elec_image = mne.warp_montage_volume(
        montage, CT_aligned, reg_affine, sdr_morph,
        subject_from=f'sub-{sub}', subject_to=template,
        subjects_dir_from=subjects_dir, subjects_dir_to=subjects_dir)
    ch_pos = montage_warped.get_positions()['ch_pos'].copy()  # use these later
    # now go back to "head" coordinates to save to raw
    montage_warped.apply_trans(mne.transforms.invert_transform(template_trans))
    raw.set_montage(montage_warped, on_missing='warn')
    mne.io.write_info(
        op.join(subjects_dir, f'sub-{sub}', 'ieeg',
                f'sub-{sub}_template-{template}_task-{task}_info.fif'),
        raw.info)
    nib.save(elec_image, op.join(subjects_dir, f'sub-{sub}', 'ieeg',
                                 'elec_image.mgz'))


# %%
# Plot final result
template_trans = mne.coreg.estimate_head_mri_t(template, subjects_dir)
for sub in subjects:
    info = mne.io.read_info(op.join(
        subjects_dir, f'sub-{sub}', 'ieeg', f'sub-{sub}_task-{task}_info.fif'))
    trans = mne.coreg.estimate_head_mri_t(f'sub-{sub}', subjects_dir)
    brain = mne.viz.Brain(f'sub-{sub}', subjects_dir=subjects_dir,
                          cortex='low_contrast', alpha=0.2, background='white')
    brain.add_sensors(info, trans)
    # plot warped
    info = mne.io.read_info(op.join(
        subjects_dir, f'sub-{sub}', 'ieeg',
        f'sub-{sub}_template-{template}_task-{task}_info.fif'))
    brain = mne.viz.Brain(template, subjects_dir=subjects_dir,
                          cortex='low_contrast', alpha=0.2, background='white')
    brain.add_sensors(info, template_trans)


# %%
# Save locations to BIDS
# Note: requires mne-bids version 0.11 (dev)
for sub in subjects:
    path.update(subject=str(sub))
    raw = mne_bids.read_raw_bids(path)
    info = mne.io.read_info(op.join(
        subjects_dir, f'sub-{sub}', 'ieeg', f'sub-{sub}_task-{task}_info.fif'))
    for ch in info['chs']:
        raw.info['chs'][raw.ch_names.index(ch['ch_name'])] = ch
    with raw.info._unlock():
        raw.info['dig'] = info['dig']
    montage = raw.get_montage()
    trans = mne.coreg.estimate_head_mri_t(f'sub-{sub}', subjects_dir)
    montage.apply_trans(trans)
    mne_bids.convert_montage_to_ras(montage, f'sub-{sub}', subjects_dir)
    dig_path = path.copy().update(datatype='ieeg', space='ACPC')
    mne_bids.dig._write_dig_bids(dig_path, raw, montage=montage,
                                 acpc_aligned=True, overwrite=True)
