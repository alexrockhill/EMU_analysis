import os
import os.path as op
import numpy as np
import nibabel as nib
import pandas as pd

import mne
import mne_bids

from params import DATA_DIR as data_dir
from params import BIDS_ROOT as bids_root
from params import SUBJECTS as subjects
from params import TASK as task
from params import TEMPLATE as template
from params import ATLAS as aseg

subjects_dir = op.join(bids_root, 'derivatives')
path = mne_bids.BIDSPath(root=bids_root, task=task)
out_dir = op.join(data_dir, 'derivatives')

# align CT, takes ~15 minutes per subject, no user input
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


# a few subjects didn't work and this was used to align (11 and 12)
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

# pick contact locations, requires user input
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

# initialize for saving positions and labels
subject = list()
electrode_name = list()
contact_number = list()
ch_position = list()
anat_label = list()
lut = {v: k for k, v in mne._freesurfer.read_freesurfer_lut()[0].items()}

# warp to template, takes ~15 minutes per subject, no user input
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
    aseg_img = np.array(nib.load(op.join(subjects_dir, f'sub-{sub}', 'mri',
                                         f'{aseg}.mgz')).dataobj)
    elec_img = np.array(elec_image.dataobj)
    for i, ch in enumerate(raw.ch_names):
        subject.append(sub)
        electrode_name.append(''.join([letter for letter in ch if
                                       not letter.isdigit()]).rstrip())
        contact_number.append(''.join([letter for letter in ch if
                                       letter.isdigit()]).rstrip())
        ch_position.append(ch_pos[ch])
        labels = [lut[idx] for idx in np.unique(aseg_img[elec_img == i + 1])]
        anat_label.append(','.join(labels))


x = [pos[0] for pos in ch_position]
y = [pos[1] for pos in ch_position]
z = [pos[2] for pos in ch_position]
pd.DataFrame(dict(sub=subject, elec_name=electrode_name, number=contact_number,
                  x=x, y=y, z=z, label=anat_label)).to_csv(
    op.join(out_dir, 'elec_contacts_info.tsv'), sep='\t', index=False)


# plot final result
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
