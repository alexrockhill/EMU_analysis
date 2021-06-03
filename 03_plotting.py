import os
import os.path as op
import numpy as np

import mne
import matplotlib.pyplot as plt

from img_pipe.viz import get_rois
from img_pipe.utils import load_electrodes
from img_pipe.config import ELECTRODE_CMAP

event = 'Response'
data_dir = f'./derivatives/pca_{event.lower()}_classifier'
fig_dir = f'./derivatives/plots'

if not op.isdir(fig_dir):
    os.makedirs(fig_dir)


with np.load(op.join(data_dir, 'scores.npz')) as scores:
    scores = scores


with np.load(op.join(data_dir, 'scores.npz')) as scores:
    scores = scores

elec_matrix = load_electrodes(template='cvs_avg35_inMNI152')

rois = get_rois('all', opacity=0.25)
renderer = mne.viz.backends.renderer.create_3d_figure(
    size=(1200, 900), bgcolor='w', scene=False)
mne.viz.set_3d_view(figure=renderer.figure, distance=500,
                    azimuth=None, elevation=None)
for elec_data in elec_matrix.values():
    x, y, z, group, _ = elec_data
    renderer.sphere(center=(x, y, z), color=ELECTRODE_CMAP(group)[:3],
                    scale=5)
for roi in rois:
    renderer.mesh(*roi.vert.T, triangles=roi.tri, color=roi.color,
                  opacity=roi.opacity, representation=roi.representation)
renderer.show()
