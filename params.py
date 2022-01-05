import numpy as np

RAW_DATA_DIR = '/home/alex/SwannLab/EMU_data'
BIDS_ROOT = '/home/alex/SwannLab/EMU_data_BIDS'
DATA_DIR = '/home/alex/SwannLab/EMU_analysis'
SUBJECTS = [1, 2, 5, 6, 9, 10, 11, 12]
TASK = 'SlowFast'
TEMPLATE = 'cvs_avg35_inMNI152'
N_COMPONENTS = 50
EVENTS = dict(baseline=('Fixation', -1.5, -0.5),
              null=('Fixation', -2.499, -1.5),
              event=('Response', -0.5, 0.499),
              go_event=('Go Cue', 0, 1))
ATLAS = 'aparc+aseg'
ALPHA = 0.01
LEFT_HANDED_SUBJECTS = [2, 11]
FREQUENCIES = np.concatenate(
    [np.linspace(1, 10, 10),
     np.logspace(np.log(11), np.log(250), 40, base=np.e)])


def neighbor_reference(raw, tol=0.5, verbose=True):
    """Reference raw data to the nearest neighbor or two.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw object.
    tol : float
        The tolerance in standard error that the second distance must
        be from the first in order to be referenced as well.

    Returns
    -------
    raw : mne.io.Raw
        The re-referenced raw object.
    """
    from scipy.spatial.distance import cdist
    raw.load_data()
    data = np.zeros(raw._data.shape) * np.nan
    ch_pos = np.array([ch['loc'][:3] for ch in raw.info['chs']])
    dists = cdist(ch_pos, ch_pos)
    np.fill_diagonal(dists, np.inf)
    for i in range(len(raw.ch_names)):
        min_idx, next_min_idx = np.argsort(dists[i])[:2]
        if abs(dists[i, next_min_idx] - dists[i, min_idx]) / \
                dists[i, min_idx] < tol:
            data[i] = raw._data[i] - (
                raw._data[min_idx] + raw._data[next_min_idx]) / 2
            if verbose:
                print(f'Referencing {raw.ch_names[i]} to '
                      f'{raw.ch_names[min_idx]} and '
                      f'{raw.ch_names[next_min_idx]}')
        else:
            data[i] = raw._data[i] - raw._data[min_idx]
            if verbose:
                print(f'Referencing {raw.ch_names[i]} to '
                      f'{raw.ch_names[min_idx]}')

    assert not np.isnan(data).any()
    raw._data = data
    return raw
