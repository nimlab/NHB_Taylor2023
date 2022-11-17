#!/bin/python3


import connectomics as cs
from collections import OrderedDict
from nilearn import image, input_data
from glob import glob
import numpy as np
import time
import argparse
import os
from tqdm import tqdm
import csv


def make_subjects(connectome_file_tuples, rois, masker, command, warning_flag):
    # Creates the ConnectomeSubject objects that are used in the nimlab.connectomics module
    subjects = []
    if(command == 'seed'):
        roi_dict = {}
    elif(command == 'matrix'):
        roi_dict = OrderedDict()

    else:
        raise ValueError("Unrecognized command")
    for r in rois:
        roi_masked = masker.transform(r)
        roi_dict.update([(r, roi_masked)])
    for f in connectome_file_tuples:
        sub = cs.ConnectomeSubject(f[0],f[1], roi_dict, warning_flag)
        subjects.append(sub)
    return subjects


if __name__ == "__main__":
    # Make sure we don't aren't being a pest
    os.nice(19)
    start = time.time()
    # Get arguments
    parser = argparse.ArgumentParser(description="Compute connectivity maps. Based off of LeadDBS's cs_fmri_conseed")
    parser.add_argument("-cs", metavar='connectome', help="Folder containing connectome/timecourse files", required=True)
    parser.add_argument("-r", metavar='ROI', help="CSV of ROI images", required=True)
    parser.add_argument("-bs", metavar='brain_space', help="Binary image that specifies which voxels of the brain to include in compressed timecourse files. Defaults to the MNI152_T1_2mm_brain_mask_dil.nii.gz image included with FSL. NOTE: The brain_space must FIRST be used to generate the connectome files.")
    parser.add_argument("-mout", metavar='output_mask', help="Output mask. Defaults to no masking.")
    parser.add_argument("-c", metavar='command', help="Seed or ROI matrix. Defaults to seed.", default='seed')
    parser.add_argument("-o", metavar='output', help="Output directory", default='seed', required=True)
    parser.add_argument("-w", metavar='workers', help="Number of workers. Defaults to 12", type=int, default=int(12))
    #Defaulting to True is kinda weird, but it makes more sense to have a rarely used flag disable warnings rather than having to pass booleans
    parser.add_argument("--nowarn", help="Activate flag to suppress warnings", default=True, action='store_false')
    parser.add_argument("-fout", metavar='file_output', help="Output individual files for each connectome subject. This can potentially generate vast quantities of data. USE WITH CARE!", type=str, default='')
    args = parser.parse_args()

    # Process arguments
    output_folder = args.o
    if(not os.path.exists(output_folder)):
        raise ValueError("Invalid output folder: " + output_folder)
    if(args.bs is None):
        mask_img = "./MNI152_T1_2mm_brain_mask_dil.nii.gz"
    else:
        mask_img = image.load_img(args.bs)
    brain_masker = input_data.NiftiMasker(mask_img)
    brain_masker.fit()

    roi_file_list = args.r
    roi_files = []
    flist = open(roi_file_list)
    reader = csv.reader(flist, delimiter=',')
    for f in reader:
        roi_files.append(f[0])
    flist.close()

    connectome_files_norms = glob(args.cs + "/*_norms.npy")
    connectome_files = [(glob(f.split('_norms')[0] + ".npy")[0], f) for f in connectome_files_norms]
    if (len(connectome_files) == 0):
        raise ValueError("No connectome files found")

    subjects = make_subjects(connectome_files, roi_files, brain_masker, args.c, args.nowarn)
    print("Loaded " + str(len(roi_files)) + " ROIs")
    print("Using " + str(len(connectome_files)) + " connectome files")

    if (args.c == 'seed'):
        print("Computing maps")
        avgR_fz_maps, avgR_maps, T_maps = cs.calculate_maps(subjects, args.w, args.fout)
        for key in avgR_fz_maps.keys():
            fname = key.split('/')[-1].split('.')[0] + '_AvgR_Fz.nii.gz'
            brain_masker.inverse_transform(avgR_fz_maps[key]).to_filename(output_folder + '/' + fname)
        for key in avgR_maps.keys():
            fname = key.split('/')[-1].split('.')[0] + '_AvgR.nii.gz'
            brain_masker.inverse_transform(avgR_maps[key]).to_filename(output_folder + '/' + fname)
        for key in T_maps.keys():
            fname = key.split('/')[-1].split('.')[0] + '_T.nii.gz'
            brain_masker.inverse_transform(T_maps[key]).to_filename(output_folder + '/' + fname)
    elif (args.c == 'matrix'):
        print("Computing matrices")
        avgR_fz_mat, avgR_mat, T_mat, roi_names = cs.calculate_roi_matrix(subjects, args.w)
        np.savetxt(output_folder + "/matrix_corrMx_AvgR_Fz.csv", avgR_fz_mat, delimiter=',')
        np.savetxt(output_folder + "/matrix_corrMx_AvgR.csv", avgR_mat, delimiter=',')
        np.savetxt(output_folder + "/matrix_corrMx_T.csv", T_mat, delimiter=',')
        name_file = open(output_folder + "/matrix_corrMx_names.csv", 'w+')
        for n in roi_names:
            name_file.write(n)
            name_file.write('\n')
        name_file.close()

    else:
        print("Unrecognized command")

    # Convert npy files for individual connectome output to niftis
    if(args.fout):
        print("Transforming individual npy files")
        ind_files = glob(args.fout+'/*/*.npy')
        for f in tqdm(ind_files):
            ind_npy = np.load(f)
            ind_img = brain_masker.inverse_transform(ind_npy)
            ind_nifti_fname = f.split(".")[0]
            ind_img.to_filename(ind_nifti_fname+'.nii.gz')

    end = time.time()
    elapsed = end - start
    print("Total elapsed: " + str(elapsed))
    print("Avg time per seed: " + str(elapsed/len(roi_files)))
