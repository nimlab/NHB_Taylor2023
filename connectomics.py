from __future__ import print_function
from collections import OrderedDict
from scipy.stats import ttest_1samp
import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import multiprocessing as mp
from numba import jit
import time
import os
from tqdm import tqdm


class ConnectomeSubject:
    """Object that represents a connectome subject and associated ROIs

    Since loading a connectome file into memory incurs a significant time penalty, we use the connectome subject as the unit of computation rather than the ROI.

    Attributes:
        connectome_file (str): File path to connectome subject file. We pass paths since we do not want to have to load connectome files until they are needed.
        norms (str): File path to the norms file for the connectome subject.
        rois (dict of str : ndarray): Dictionary of ROIs. The string specifies the name of the ROI (often the filename), and the ndarray is the masked and flattened ROI image.
	warning_flag (bool): Boolean to determine whether to display warning messages (True will display warnings). 
    """

    def __init__(self, connectome_file, connectome_norms, rois, warning_flag):
        self.connectome_file = connectome_file
        self.norms = connectome_norms
        self.rois = rois
        self.warning_flag = warning_flag


def make_fz_maps(subject):
    """ Calculates the Fischer transformed average Pearson R correlation between a roi region and the rest of the voxels in a
    connectome file.

    Args:
        subject (ConnectomeSubject): The connectome subject file (with associated ROIs) to compute fz maps for.

    Returns:
       (dict of str : ndarray): Dictionary mapping the ROI name to its fz map
    """
    connectome_mat = np.load(subject.connectome_file).astype(np.float32)
    connectome_norms_mat = np.load(subject.norms).astype(np.float32)
    fz_maps = {}
    for roi_key in subject.rois.keys():
        roi = subject.rois[roi_key]
        if(connectome_mat.shape[1] != roi.shape[1]):
            raise ValueError("ROI not masked with same mask as connectome file. \
                    Connectome file has " + str(connectome_mat.shape[1]) + " voxels, while \
                    the ROI has " + str(roi.shape[1]))

        # Mask connectome time courses, find mean tc of masked area
        roi_mean_tc = extract_avg_signal(connectome_mat, roi)

        # Perform correlation
        corr_num = np.dot(connectome_mat.T, roi_mean_tc)
        corr_denom = connectome_norms_mat*np.linalg.norm(roi_mean_tc)

        np.seterr(invalid='ignore')
        corr = corr_num / corr_denom
        if(subject.warning_flag):
            if(corr.max() > 1):
                print(roi_key)
                print("Unexpected corr value: " + str(corr.max()))

        # Make sure fisher z transform is taking in valid values
        corr[np.isnan(corr)] = 0

        fz = np.arctanh(corr)

        # Fix infinite values in the case of single voxel autocorrelations
        finite_max = fz[np.isfinite(fz)].max()
        fz[np.isinf(fz)] = finite_max

        fz_maps.update([(roi_key, fz)])

    return fz_maps


def extract_avg_signal(connectome_mat, roi_mat):
    """Extracts a single time course from a region in a connectome file specified by an ROI

    The current extraction method is to mask out all voxels where the ROI image is 0, multiply the remaining connectome
    voxs by the weights specified by the ROI image, and then average all voxel timecourses together.
    NOTE: The ROI image can be weighted, but should be all positive.

    Args:
        connecotme_mat(ndarray): Masked and flattened connecotme subject image
        roi_mat(ndarray): Masked and flattened ROI image of the same length as the connectome subject

    Returns:
        ndarray: Extracted signal. Has same shape as the connecotme file and roi image
    """
    roi_masked_tc = connectome_mat[:, roi_mat[0, :] > 0]
    roi_masked_tc = roi_masked_tc * roi_mat[roi_mat[0:] > 0]
    roi_mean_tc = np.nanmean(roi_masked_tc, axis=1)

    return roi_mean_tc


def make_fz_maps_to_queue(subject, result_queue, ind_file_output=''):
    connectome_fname = subject.connectome_file.split('/')[-1]
    fz_maps = make_fz_maps(subject)
    for roi_key in fz_maps.keys():
        result_queue.put((roi_key, fz_maps[roi_key]))
        if(ind_file_output):
            # Probably making some unsafe assumptions about ROI naming, but
            # I don't see any better way to do it.
            roi_name = roi_key.split('/')[-1].split('.')[0]
            roi_ind_directory = ind_file_output + '/' + roi_name
            if not os.path.exists(roi_ind_directory):
                os.makedirs(roi_ind_directory)
            out_fname = roi_ind_directory + '/' + connectome_fname
            np.save(out_fname, fz_maps[roi_key])


def gen_welford_maps_from_queue(n_connectome, n_roi, result_queue, welford_maps_queue):
    """Creates welford maps from a queue filled with fz maps from make_fz_maps()

    Args:
        n_connectome(int): Number of connectome files
        n_roi(int): Number of ROIs
        result_queue(Queue): Queue of fz maps
        welford_maps_queue(Queue): Queue onto which welford maps generated are pushed
    """
    current_count = 0
    welford_maps = {}
    total_maps = n_connectome * n_roi
    with tqdm(total=total_maps) as pbar:
        while (current_count < total_maps):
            roi_result = result_queue.get()
            roi_key = roi_result[0]
            roi_map = roi_result[1]
            if(roi_key not in welford_maps.keys()):
                fz_welford_result = welford_update_map(np.zeros(roi_map.shape[0], dtype=(np.float32, 3)),
                                                       roi_map)
            else:
                fz_welford_result = welford_update_map(welford_maps[roi_key], roi_map)
            welford_maps.update([(roi_key, fz_welford_result)])
            pbar.update(1)
            current_count += 1
    print("Welford maps constructed")
    for m in welford_maps.items():
        welford_maps_queue.put(m)


def calculate_maps(subjects, num_workers, ind_file_output=''):
    """ Calculates AvgR_Fz, AvgR, and T maps for a set of subjects

    Args:
        subjects(ConnectomeSubject[]): List of connectome subjects with associated ROIs
        num_workers(int): Number of workers processes

    Returns:
        (dict, dict, dict): Dictionaries of AvgR_Fz maps, AvgR maps, T maps
    """
    # TODO: refactor tuples scheme, since we are no longer using pool/map
    os.nice(19)
    result_queue = mp.Queue()
    final_maps_queue = mp.Queue()
    map_maker = mp.Process(target=gen_welford_maps_from_queue, args=(len(subjects),
                           len(subjects[0].rois), result_queue, final_maps_queue))
    map_maker.start()
    num_pool_workers = num_workers
    print("Using pool of " + str(num_pool_workers) + " workers")
    pool = []
    worker_queue = []
    for s in subjects:
        w = mp.Process(target=make_fz_maps_to_queue, args=(s, result_queue, ind_file_output))
        worker_queue.append(w)
    for job in worker_queue:
        while(len(pool) >= num_pool_workers):
            for w in pool:
                if(w.is_alive() is False):
                    pool.remove(w)
            time.sleep(1)
        pool.append(job)
        job.start()
    fz_welford_maps = {}
    for i in range(0, len(subjects[0].rois)):
        welford_map = final_maps_queue.get()
        fz_welford_maps.update([welford_map])
    map_maker.join()

    avgR_fz_maps, avgR_maps, T_maps = make_stat_maps(fz_welford_maps, len(subjects))

    return avgR_fz_maps, avgR_maps, T_maps


def make_stat_maps(fz_welford_maps, num_subjects):
    """ Generates statistical maps from welford maps

    Args:
        fz_welford_maps(dict of str : ndarray): Dictionary mapping ROI names to welford maps

    Returns:
        (dict, dict, dict): avgR_fz maps, avgR_maps, T_maps
    """
    avgR_fz_maps = {}
    avgR_maps = {}
    T_maps = {}
    print("Constructing statistical maps")
    for roi_key in tqdm(fz_welford_maps.keys()):
        fz_welford_final = np.asarray(welford_finalize_map(fz_welford_maps[roi_key]))

        # construct avgR_fz map
        avgR_fz = fz_welford_final[:, 0]
        avgR_fz_maps.update([(roi_key, avgR_fz)])

        # construct avgR map
        avgR = np.tanh(avgR_fz)
        avgR_maps.update([(roi_key, avgR)])

        # construct T map
        variance = fz_welford_final[:, 2]
        num_samples = num_subjects
        t_test_denom = np.sqrt(variance / num_samples)
        T = avgR_fz / t_test_denom
        T_maps.update([(roi_key, T)])

    return avgR_fz_maps, avgR_maps, T_maps


def calculate_roi_matrix(subjects, num_workers):
    """Calculates and ROI-ROI matrix correlation for a set of connectome subjects (with associated ROIs)

    Args:
        subjects(ConenctomeSubject[]): set of subjects with ROIs
        num_workers(int): Number of worker processes

    Returns:
        ndarray, ndarray, ndarray, roi_names: AvgR_Fz matrix, AvgR matrix, T matrix, list of ROI names in order
    """
    os.nice(19)
    pool = Pool(num_workers)
    fz_list = list(tqdm(pool.imap(make_fz_matrix, subjects), total=len(subjects)))
    fz_array = np.asarray(fz_list)
    avgR_fz = np.mean(fz_array, axis=0)
    avgR = np.tanh(avgR_fz)
    T = ttest_1samp(fz_array, 0, axis=0)[0]

    roi_names = subjects[0].rois.keys()

    return avgR_fz, avgR, T, roi_names


def make_fz_matrix(subject):
    """Creates a fz transformed matrix of ROI-ROI correlations for a single subject

    Args:
        subject(ConnectomeSubject): Subject with associated ROIs

    Returns:
        ndarray: R_Fz matrix
    """
    roi_tcs = OrderedDict()
    connectome_mat = np.load(subject.connectome_file)
    for roi_key in subject.rois.keys():
        roi_tcs.update([(roi_key, extract_avg_signal(connectome_mat, subject.rois[roi_key]))])
    corrs = np.corrcoef(list(roi_tcs.values()))
    corrs[np.isnan(corrs)] = 0
    np.seterr(divide='ignore')
    fz = np.arctanh(corrs)

    return fz


@jit(nopython=True)
def welford_update_map(existingAggregateMap, newMap):
    newAggregates = []
    for i in range(0, existingAggregateMap.shape[0]):
        newAggregate = welford_update(existingAggregateMap[i], newMap[i])
        newAggregates.append(newAggregate)

    return np.asarray(newAggregates)

# The following two methods are from wikipedia:
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
@jit(nopython=True)
def welford_update(existingAggregate, newValue):
    (count, mean, M2) = existingAggregate
    count += 1
    delta = newValue - mean
    mean += delta / count
    delta2 = newValue - mean
    M2 += delta * delta2

    return (count, mean, M2)


@jit(nopython=True)
def welford_finalize_map(existingAggregateMap):
    finalMap = []
    for i in range(0, existingAggregateMap.shape[0]):
        finalized_map = welford_finalize(existingAggregateMap[i])
        finalMap.append(finalized_map)
    return finalMap

# retrieve the mean, variance and sample variance from an aggregate
@jit(nopython=True)
def welford_finalize(existingAggregate):
    (count, mean, M2) = existingAggregate
    (mean, variance, sampleVariance) = (mean, M2/count, M2/(count - 1))
    return np.asarray([mean, variance, sampleVariance], dtype=np.float32)
