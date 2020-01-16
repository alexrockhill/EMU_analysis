from swann.utils import get_config, get_layout, exclude_subjects
from swann.preprocessing import preproc_slowfast, slowfast_group_stats
from swann.viz import plot_slow_fast, plot_slow_fast_group

config = get_config()
layout = get_layout()

behfs = layout.get(task=config['task'],
                   extension='tsv', suffix='beh')
behfs = exclude_subjects(behfs)

overwrite_beh = \
    input('Overwrite behavioral data if they exist? (y/n)\n').upper() == 'Y'

overwrite_plots = \
    input('Overwrite plots if they exist? (y/n)\n').upper() == 'Y'

# loop across subjects
for behf in behfs:
    preproc_slowfast(behf, overwrite=overwrite_beh)
    fig = plot_slow_fast(behf, overwrite=overwrite_plots)

plot_slow_fast_group(behfs, 'EMU Patients', overwrite=overwrite_plots)
slowfast_group_stats(behfs, 'EMU Patients', overwrite=overwrite_beh)
