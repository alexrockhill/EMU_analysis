# EMU_analysis

Procedure for importing data/converting to BIDS
1. Convert the .mat data file to a tsv using slowfast_mat2csv.m or another task-specific conversion
2. Find the photodiode events with swann/bids/parse_pd.py
    command line argument example: python parse_pd.py --eegf EMU_data/sub-1_raw/XXXXX.edf
                                                      --behf EMU_data/sub-1_raw/slowfast_sub-1.tsv
    Note columns must align with default arguments (only works for slowfast) otherwise arguments
    --relative_event_cols and --relative_event_names must be supplied
    This should work if the photodiodes are recoverable by eye
3. Save to bids using swann/bids/save2bids.py
    command line argument example: python save2bids.py --bids_dir EMU_data_BIDS/ --sub 1
                                                       --task SlowFast
                                                       --eegf EMU_data/sub-1_raw/XXXXX.edf
                                                       --data_ch_type seeg
                                                       --task SlowFast
4. Convert MRI and CT from DICOM to nii.gz
    If you don't have it, install from here https://people.cas.sc.edu/rorden/mricron/install.html and add to path

    example bash commands:
    cd EMU_data/sub-1_raw
    mkdir MRI_nii
    dcm2niix -o ./MRI_nii -z y ./MRI/
    freeview MRI_nii/the_good_one_T1.nii.gz  # you might have to look though them all if you don't know which is good
    cp MRI_nii/the_good_one_T1.nii.gz ../../EMU_data_BIDS/sub-1/anat/sub-1_T1w.nii.gz # can do also for T2/FLAIR for source localization analyses esp if EEG simulatenously
    cp MRI_nii/the_good_one_T1.json ../../EMU_data_BIDS/sub-1/anat/sub-1_T1w.json

    mkdir CT_nii
    dcm2niix -o ./CT_nii -z y ./CT/
    freeview MRI_nii/the_good_one_CT.nii.gz
    cp MRI_nii/the_good_one_CT.nii.gz ../../EMU_data_BIDS/sub-1/ct/sub-1_ct.nii.gz
    cp MRI_nii/the_good_one_CT.json ../../EMU_data_BIDS/sub-1/ct/sub-1_ct.json

5. Localize the electrodes using img_pipe (see instructions from package and Frotiers paper https://www.frontiersin.org/articles/10.3389/fninf.2017.00062/full )

# TO DO: make install easier
