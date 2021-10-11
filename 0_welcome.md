# Welcome
Welcome to the analysis of intracranial data from epilepsy patients at Oregon Health & Science University written by Alex Rockhill (arockhil@uoregon.edu) at the University of Oregon department of Human Physiology. 

At this point to replicate and walk through the analysis, you will need:
1) To install python, using Anaconda is recommended (https://www.anaconda.com/distribution/#download-section)
2) Install MNE including the development environment, see https://mne.tools/dev/install/mne_python.html
3) the Github Respository for this analysis (https://github.com/alexrockhill/EMU_analysis)
   a) change directory to where you want to put the project
   b) run `git clone https://github.com/alexrockhill/EMU_analysis`
4) the BIDS formatted data on OpenNeuro (link)

Then you can run each python file in order, e.g. `python 1_gui.py`

For information on how this dataset was converted to BIDS, see https://mne.tools/mne-bids/dev/auto_examples/convert_ieeg_to_bids.html

# References
Yarkoni et al., (2019). PyBIDS: Python tools for BIDS datasets. Journal of Open Source Software, 4(40), 1294, https://doi.org/10.21105/joss.01294

Yarkoni, Tal, Markiewicz, Christopher J., de la Vega, Alejandro, Gorgolewski, Krzysztof J., Halchenko, Yaroslav O., Salo, Taylor, ...,  Blair, Ross. (2019, August 8). bids-standard/pybids: 0.9.3 (Version 0.9.3). Zenodo. http://doi.org/10.5281/zenodo.3363985

Mainak Jas, Eric Larson, Denis Engemann, Jaakko Leppakangas, Samu Taulu, Matti Hamalainen, and Alexandre Gramfort. 2018. A Reproducible MEG/EEG Group Study With the MNE Software: Recommendations, Quality Assessments, and Good Practices. Frontiers in Neuroscience. 12, doi: 10.3389/fnins.2018.00530
