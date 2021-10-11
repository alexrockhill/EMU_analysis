import os
import os.path as op
import nibabel as nib
import numpy as np
import mne
import mne_bids
from params import BIDS_ROOT as bids_root

subjects_dir = op.join(bids_root, 'derivatives')
subjects = [1, 2, 5, 6, 9, 10, 11, 12]
template = 'cvs_avg35_inMNI152'
path = mne_bids.BIDSPath(root=bids_root, task='SlowFast')


# align CT
for sub in subjects:
    T1 = nib.load(op.join(subjects_dir, f'sub-{sub}', 'mri', 'T1.mgz'))
    CT_orig = nib.load(op.join(subjects_dir, f'sub-{sub}', 'CT', 'CT.nii'))
    reg_affine, _ = mne.transforms.compute_volume_registration(
        CT_orig, T1, pipeline='rigids')
    CT_aligned = mne.transforms.apply_volume_registration(
        CT_orig, T1, reg_affine)
    nib.save(CT_aligned, op.join(
        subjects_dir, f'sub-{sub}', 'CT', 'CT_aligned.mgz'))


# pick contact locations
for sub in subjects:
    path.update(subject=str(sub))
    raw = mne_bids.read_raw_bids(path)
    raw.set_montage(None)
    CT_aligned = nib.load(op.join(
        subjects_dir, f'sub-{sub}', 'CT', 'CT_aligned.mgz'))
    trans = mne.coreg.estimate_head_mri_t(f'sub-{sub}', subjects_dir)
    gui = mne.gui.locate_ieeg(raw.info, trans, CT_aligned,
                              subject=f'sub-{sub}', subjects_dir=subjects_dir)
    while input('Finished, save to disk? (y/N)\t') != 'y':
        mne.io.write_info(op.join(subjects_dir, f'sub-{sub}', 'ieeg',
                                  f'sub-{sub}_task-{path.task}_info.fif'),
                          raw.info)

# warp to template
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
        subjects_dir, f'sub-{sub}', 'ieeg',
        f'sub-{sub}_task-{path.task}_info.fif'))
    raw.drop_channels([ch for ch in raw.ch_names if ch not in info.ch_names])
    raw.info = info
    trans = mne.coreg.estimate_head_mri_t(f'sub-{sub}', subjects_dir)
    CT_aligned = nib.load(op.join(
        subjects_dir, f'sub-{sub}', 'CT', 'CT_aligned.mgz'))
    subject_brain = nib.load(
        op.join(subjects_dir, f'sub-{sub}', 'mri', 'brain.mgz'))
    reg_affine, sdr_morph = mne.transforms.compute_volume_registration(
        subject_brain, template_brain, verbose=True)
    montage = raw.get_montage()
    montage.apply_trans(trans)
    montage_warped, elec_image, warped_elec_image = mne.warp_montage_volume(
        montage, CT_aligned, reg_affine, sdr_morph,
        subject_from=f'sub-{sub}', subject_to=template,
        subjects_dir=subjects_dir)
    montage_warped.apply_trans(mne.transforms.invert_transform(template_trans))
    raw.set_montage(montage_warped, on_missing='warn')
    mne.io.write_info(
        op.join(subjects_dir, f'sub-{sub}', 'ieeg',
                f'sub-{sub}_template-{template}_task-{path.task}_info.fif'),
        raw.info)


# plot final result
for sub in subjects:
    info = mne.io.read_info(op.join(
        subjects_dir, f'sub-{sub}', 'ieeg',
        f'sub-{sub}_task-{path.task}_info.fif'))
    trans = mne.coreg.estimate_head_mri_t(f'sub-{sub}', subjects_dir)
    brain = mne.viz.Brain(f'sub-{sub}', subjects_dir=subjects_dir,
                          cortex='low_contrast', alpha=0.2, background='white')
    brain.add_sensors(info, trans)
    # plot warped
    info = mne.io.read_info(op.join(
        subjects_dir, f'sub-{sub}', 'ieeg',
        f'sub-{sub}_template-{template}_task-{path.task}_info.fif'))
    brain = mne.viz.Brain(template, subjects_dir=subjects_dir,
                          cortex='low_contrast', alpha=0.2, background='white')
    brain.add_sensors(info, template_trans)
