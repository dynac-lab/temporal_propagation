import mne
import pandas as pd
import mne_bids
import config as cfg
import matplotlib.patches as patches
from tqdm import tqdm
import nibabel
import random
from statesegmentation import GSBS
import matplotlib.pyplot as plt
import re
from fractions import Fraction
from scipy.signal import resample_poly
import numpy as np


def get_electrode_coordinates(n, chan_names=None, include_names=True):
    # Load electrodes
    channels_path = mne_bids.BIDSPath(subject=str(n).zfill(2),
                                      session='iemu',
                                      suffix='electrodes',
                                      extension='.tsv',
                                      datatype='ieeg',
                                      acquisition='clinical',
                                      root=cfg.dir_data)
    electrodes = pd.read_csv(str(channels_path.match()[0]), sep='\t', header=0, index_col=None)

    # set chan names if not existent
    if chan_names is None:
        chan_names = electrodes['name'].values

    # Get coordinates and transform to RAS
    coordinates_all = electrodes[['x', 'y', 'z']].values
    x = nibabel.load(cfg.dir_fs +  'sub-' + str(n).zfill(2) + '/mri/orig.mgz')
    vox_coords = np.round(mne.transforms.apply_trans(np.linalg.inv(x.affine), coordinates_all)).astype(int)
    ras_coords_electrodes = mne.transforms.apply_trans(x.header.get_vox2ras_tkr(), vox_coords)
    ras_coords_electrodes = ras_coords_electrodes / 1000


    if include_names:
        # Create dictionary: chan_name, coordinates
        chan_dict = {}
        for chan_name in chan_names:
            idx = [np.where(electrodes['name'].values == chan_name)[0][0]]
            chan_dict[chan_name] = ras_coords_electrodes[idx, :].flatten()
        return chan_dict
    else:
        chan_array = [ras_coords_electrodes[idx, :].flatten() for idx in range(ras_coords_electrodes.shape[0]) if electrodes['name'][idx] in chan_names]
        return np.asarray(chan_array)



def plot_time_correlation(ax, data, GSBS=None, nstates=None):
    # ax: where it is plotted
    # data: 2D matrix, time x voxels
    # GSBS (opt): GSBS object that has been fit

    # Compute corrcoef
    corr = np.corrcoef(data)

    # Plot the matrix
    ax.imshow(corr, interpolation='none', vmin=-1, vmax=1)
    ax.set_xlabel('Timepoint')
    ax.set_ylabel('Timepoint')

    # Plot the boundaries
    if GSBS is not None:
        strengths = GSBS.get_strengths(nstates)
        bounds = np.where(strengths > 0)[0]

        # Add start and end
        bounds = np.insert(bounds, 0, 0)
        bounds = np.append(bounds, len(strengths))

        for i in range(len(bounds) - 1):
            rect = patches.Rectangle(
                (bounds[i], bounds[i]),
                bounds[i + 1] - bounds[i],
                bounds[i + 1] - bounds[i],
                linewidth=1, edgecolor='w', facecolor='none'
            )
            ax.add_patch(rect)

def get_start_and_end_stimulus(raw):
    custom_mapping = {'Stimulus/music': 2, 'Stimulus/speech': 1,
                      'Stimulus/end task': 5}  # 'Stimulus/task end' in laan
    events, event_id = mne.events_from_annotations(raw, event_id=custom_mapping, use_rounding=False)
    stim_start = events[0, 0]
    stim_end = events[-1, 0]
    return stim_start, stim_end

def extract_block_withoutEventInfo(data, block_nr, fs):
    # data: space x time
    t_start = block_nr * fs * 30
    t_end = (block_nr + 1) * fs * 30
    return data[:, t_start: t_end]

def extract_blocks(raw):
    data_all = raw.get_data()

    data_blocks = {}
    custom_mapping = {'Stimulus/music': 2, 'Stimulus/speech': 1,
                      'Stimulus/end task': 5}  # 'Stimulus/task end' in laan
    events, event_id = mne.events_from_annotations(raw, event_id=custom_mapping, use_rounding=False)
    timepoints = events[:,0]

    # Check assumption
    assert (np.sum(events[:,2] == [2,1,2,1,2,1,2,1,2,1,2,1,2,5]) != 14)

    for t in range(13):
        data_blocks[t] = data_all[:,timepoints[t]: timepoints[t+1]]
    return data_blocks

def plot_tdistance_over_states(ax, GSBS):
    ax.plot(GSBS.tdists)
    ax.scatter(GSBS.nstates, GSBS.tdists[GSBS.nstates], marker='o', facecolor='r', edgecolors='k')
    ax.set_xlabel('Number of states')
    ax.set_ylabel('T-distance')

def zscore(x): # Copy from GSBS, thus x should be time by vox to make it similar to GSBS
    return (x - x.mean(1, keepdims=True)) / x.std(1, keepdims=True, ddof=1)

def clip(data):
    # Compute SD per electrode
    for el_idx in range(data.shape[0]):
        SD = np.std(data[el_idx,:])
        avg = np.mean(data[el_idx,:])
        a_min = avg-3*SD
        a_max = avg+3*SD
        data[el_idx,:] = np.clip(data[el_idx,:], a_min=a_min, a_max=a_max)

    return data

def gaussian(x, mu, sig):
    distribution = 1./(np.sqrt(2.*np.pi) * sig) * np.exp(-np.power((x-mu)/sig, 2.)/2)
    return distribution/np.max(distribution)

def match_per_run(bounds_seed, bounds_other):
    # Note: bounds_seed and bounds_other must be binary
    # bounds_seed/bounds_other: dictionary with block/run numbers as keys. The values per item are np arrays.

    # Get a Gaussian
    x = np.arange(-int(cfg.GSBS_fs * 10), int(cfg.GSBS_fs * 10)) # In number of timepoints
    gaussian_window = gaussian(x, 0, cfg.gaussian_sd)

    match_sum = 0
    nr_of_bounds_seed = 0

    # Loop over blocks
    for block in bounds_seed.keys():
        # Keep track of the number of bounds
        nr_of_bounds_seed += np.sum(bounds_seed[block])

        # Loop over boundaries in the seed timeline of this block
        for bound_seed in np.where(bounds_seed[block])[0]:
            # Get differences between the one boundary seed and all boundaries in bounds_other
            diffs = np.abs(np.where(bounds_other[block])[0] - bound_seed)

            # Get the best difference
            diff_closest = np.min(diffs)

            # Get and store weighted match from the Gaussian
            match_sum += gaussian_window[np.where(x == diff_closest)[0][0]]

    # Weigh the sum by the number of boundaries in the seed
    return match_sum/nr_of_bounds_seed


def permutation_test_per_run(timeline_permuting, func_statistic, tailed, show_progress=False, **kwargs):
    # Perform a permutation test with each permutation having a different order of states, but with the states still being of the same length
    # timeline_permuting: dictionary, with keys being runs, and values being a vector with strength values per timepoint, with 0 denoting the absence of a boundary.
    # func_statistic: function to compute the statistic
    # tailed: 'two-tailed', 'smaller', 'bigger'
    # *arg: any argument that has to be given to func_statistic, on top of timeline_permuting

    stat_data = func_statistic(timeline_permuting, **kwargs)
    num_perms = cfg.nr_permutations


    stat_null = dict((key,np.ones((num_perms))) for key in stat_data.keys())

    if show_progress:
        to_loop = tqdm(range(num_perms))
    else:
        to_loop = range(num_perms)

    for iteration in to_loop:
        strength_shuffled = shuffle_states_per_run(timeline_permuting)
        stat_iteration = func_statistic(strength_shuffled,  **kwargs)
        for key in stat_data.keys():
            stat_null[key][iteration] = stat_iteration[key]


    ps = {}
    mean_of_null = {}
    sd_of_null = {}
    for key in stat_data.keys():
        if tailed == 'smaller':
            ps[key] = np.sum(stat_null[key] <= stat_data[key]) / num_perms
        elif tailed == 'bigger':
            ps[key] = np.sum(stat_null[key] >= stat_data[key]) / num_perms
        elif tailed == 'two-tailed':
            Exception("Not implemented yet")
        mean_of_null[key] = np.mean(stat_null[key])
        sd_of_null[key] = np.std(stat_null[key])

    return ps, mean_of_null, sd_of_null

def shuffle_states(timeline):
    # This shuffling function words on non-binary state timelines (i.e., boundary may have a strength value)

    # Get new order of state lengths
    state_order = np.arange(0, np.sum(timeline > 0) + 1)
    random.shuffle(state_order)

    # Get original start indices of each state
    state_start_org = np.where(timeline > 0)[0]
    state_start_org = np.insert(state_start_org, 0, 0)

    # Get strengths and re-order them
    strengths_shuffled = timeline[timeline>0]
    random.shuffle(strengths_shuffled)

    timeline_shuffled = []
    for strength_idx, state_idx in enumerate(state_order):
        # find start and end idx over time
        start = state_start_org[state_idx]
        if state_idx == np.max(state_order):
            end = len(timeline)
        else:
            end = state_start_org[state_idx + 1]
        state_length = end-start

        # If this is not the first state in the timeline, add the next strength
        if strength_idx != 0:
            timeline_shuffled.append(strengths_shuffled[strength_idx-1])

            # Add the length to shuffled timeline
            timeline_shuffled.extend(np.zeros(state_length -1))
        else: # Add length without any strength
            timeline_shuffled.extend(np.zeros(state_length))

    return np.array(timeline_shuffled)

def shuffle_states_per_run(timeline_per_run):
    timeline_per_run_shuffled = {}
    for run_nr in timeline_per_run.keys():
        timeline_per_run_shuffled[run_nr] = shuffle_states(timeline_per_run[run_nr])
    return timeline_per_run_shuffled

def timepoints_to_timeline(timepoints, fs, duration):
    # duration: in seconds
    timeline = np.zeros(duration * fs)
    timeline[(timepoints*fs).astype(int)] = 1
    return timeline

def run_GSBS(x, savename):
    # x: space x time
    GSBS_obj = GSBS(x=x.T, kmax=int(0.5 * x.shape[1]), statewise_detection=True)
    GSBS_obj.fit()
    np.save(savename, GSBS_obj)

def get_durations_from_GSBS_obj(GSBS_obj):
    deltas = GSBS_obj.deltas
    durations = []
    count = 0
    for d in deltas:
        if d:
            durations.append(count)
            count = 1
        else:
            count += 1
    return np.asarray(durations)

def get_durations_from_timelines(timelines):
    durations = []
    for block in timelines.keys():
        count = 0 # Rest counter at the start of every block
        deltas = timelines[block] > 0
        for d in deltas:
            if d:
                durations.append(count) # add current count, excluding this timepoint
                count = 1 # not zero as current timepoint is already part of next state
            else:
                count += 1
    return np.asarray(durations)

def add_delay(timeline, delay, fs, cut_end = True):
    # timeline: array, electrode x time, or 1 x time
    # delay: float in seconds

    timeline_delayed = timeline.copy()
    nr_of_extra_values = np.abs(int(delay * fs))

    if len(timeline.shape) == 1:  # 1D array
        axis = 0
        extra_values = np.zeros(nr_of_extra_values, dtype=int)
        nr_of_timepoints = len(timeline)
    else:  # 2D array
        axis = 1
        extra_values = np.zeros((nr_of_extra_values, timeline.shape[0]), dtype=int)
        nr_of_timepoints = timeline.shape[1]

    if delay >= 0:

        # Add zeros at the start
        timeline_delayed = np.insert(timeline_delayed, 0, extra_values, axis=axis)

        # Cut extra values
        if cut_end:
            if len(timeline.shape) == 1: # 1D array
                timeline_delayed = timeline_delayed[:nr_of_timepoints]
            else:
                timeline_delayed = timeline_delayed[:, :nr_of_timepoints]
    else:
        if len(timeline_delayed.shape) > 1:
            raise Exception("Negative delay for 2D array not implemented (yet)")

    return timeline_delayed

def p_to_str(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""

def get_relative_match(match, null_value):
    return (match - null_value)/(1-null_value)

def plot_line_with_shade(x,y,yerror, **kwargs):
    plt.plot(x,y, '-k')
    plt.fill_between(x, y-yerror, y+yerror, **kwargs)

def get_peaks(y):
    # X is a peak when on of the below holds:
    # 1) y(X-1) and y(X+1) are smaller than or equal to y(X)
    # 2) if y(x-1) == y(x), and y(x-2) is smaller than y(x)
    # 3) if y(x+1) == y(x), and y(x+2) is smaller than y(x)

    peaks = []

    for x in range(1,len(y)-1):
        left = False
        right = False

        # Check left
        if y[x-1] < y[x]:
            left = True
        if y[x-1] == y[x]:
            if x-2 < 0 or y[x-2] < y[x]:
                left = True

        # Check right
        if y[x+1] < y[x]:
            right = True
        if y[x+1] == y[x]:
            if x+2 >= len(y) or y[x+2] < y[x]:
                right = True

        if left and right:
            peaks.append(x)
    return peaks

def zeropad_tdistances(t_distances):
    # t_distances: list of arrays.
    # This function checks if the arrays in t_distances are of different lengths, and zero-pads them at the end if necessary
    max_length = 0
    for dists in t_distances:
        max_length = np.max((max_length, len(dists)))

    for block_idx in range(len(t_distances)):
        while len(t_distances[block_idx]) < max_length:
            t_distances[block_idx] = np.append(t_distances[block_idx],0)
    return t_distances

def load_adjusted_strengths(subject, ROI, blocks):
    strengths = {}
    for block in blocks:
        filename = cfg.dir_GSBS + "GSBS_" + subject + "_" + cfg.band + "_ROI-" + ROI + "_block" + str(block) + ".npy"
        GSBS_obj = np.load(filename, allow_pickle=True).item()
        strengths[block] = GSBS_obj.get_strengths(cfg.adjusted_nr_of_states[ROI][subject][block])
    return strengths


### The functions below are taken from/based on the example code provided by Berezutskaya et al. 2022 ###
def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    """
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )

def resample(x, sr1, sr2, axis=0):
    '''sr1: target, sr2: source'''
    a, b = Fraction(sr1, sr2)._numerator, Fraction(sr1, sr2)._denominator
    return resample_poly(x, a, b, axis).astype(np.float32)

def smooth_signal(y, n):
    box = np.ones(n)/n
    ys = np.convolve(y, box, mode='same')
    return ys
