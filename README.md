# Welcome
Welcome to the analysis of intracranial data from epilepsy patients at Oregon Health & Science University written by Alex Rockhill (arockhil@uoregon.edu) at the University of Oregon department of Human Physiology. 

At this point to replicate and walk through the analysis, you will need to:
1) Install Python (https://www.python.org/downloads/)
2) Install MNE including the development environment, see https://mne.tools/dev/install/mne_python.html
3) Download the Github Respository for this analysis (https://github.com/alexrockhill/EMU_analysis)
   a) Change your working directory to where you want to put the project
   b) Run `git clone https://github.com/alexrockhill/EMU_analysis` in a command window/terminal
4) Download the BIDS formatted data on OpenNeuro (https://openneuro.org/datasets/ds004085)

Then you can run each python file in order, e.g. `python 1_preprocess.py` (you might want to run them section by section but they will execute start to finish)

For information on how this dataset was converted to BIDS, see https://mne.tools/mne-bids/dev/auto_examples/convert_ieeg_to_bids.html

Note that the original magnetic resonance (MR) images are not aligned to anterior commissuire-posterior commissure (ACPC) but the T1.mgz in the Freesurfer subject's directory is, this is T1 defines the coordinate system of the electrode locations.
