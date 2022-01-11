import numpy as np
import mne

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


def bipolar_reference(raw, verbose=True):
    """Reference raw data bipolar.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw object.

    Returns
    -------
    raw : mne.io.Raw
        The re-referenced raw object.
    """
    raw.load_data()
    ch_names = [name.replace(' ', '') for name in raw.ch_names]  # no spaces
    bipolar_names = list()
    locs = list()
    data = list()
    chs_used = list()
    for i, ch in enumerate(ch_names):
        if ch in chs_used:
            continue
        elec_name = ''.join([letter for letter in ch if
                             not letter.isdigit()]).rstrip()
        number = ''.join([letter for letter in ch if
                          letter.isdigit()]).rstrip()
        pair = f'{elec_name}{int(number) + 1}'
        if pair in chs_used or pair not in ch_names:
            continue
        j = ch_names.index(pair)
        data.append(raw._data[i] - raw._data[j])
        locs.append((raw.info['chs'][i]['loc'][:3] +
                     raw.info['chs'][j]['loc'][:3]) / 2)
        chs_used.append(ch)
        chs_used.append(pair)
        bipolar_names.append(f'{ch}-{pair}')
        if verbose:
            print(f'Bipolar referencing {ch} and {pair}')
    bipolar_info = mne.create_info(bipolar_names, raw.info['sfreq'], 'seeg')
    for loc, ch in zip(locs, bipolar_info['chs']):
        ch['loc'][:3] = loc
    return mne.io.RawArray(np.array(data), bipolar_info)
