import glob
import multiprocessing
import numpy as np
from nilearn import image, maskers
from tqdm import tqdm
from natsort import natsorted
import argparse
import os



# Function that transforms a set of nifti files to an npy file and a norms file for use with connectome_quick
# args is a tuple in the form (files[], subject_name, mask_file, output_dir)
def transform(args):
    files = args[0]
    subject_name = args[1]
    mask_file = args[2]
    output_dir = args[3]
    subject_img = image.concat_imgs(files)
    masker = maskers.NiftiMasker(mask_file, standardize=False)
    masked = masker.fit_transform(subject_img)
    norms = np.linalg.norm(masked, axis = 0)
    np.save(os.path.join(output_dir,subject_name),masked.astype('float16'))
    np.save(os.path.join(output_dir,subject_name+'_norms'),norms.astype('float16'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Format 4d Nifti files into 2d npy files with associated pre computed norms"
    )
    parser.add_argument('input')
    parser.add_argument('output')
    args = parser.parse_args()
    input_dir = args.input
    output_dir = args.output

    # All NIMLAB analyses have used the FSL 2mm_brain_mask_dil file since shifting to python code (older analyses used the 222.nii.gz mask as in Lead-DBS)
    mask_img = "./MNI152_T1_2mm_brain_mask_dil.nii.gz"

    files = natsorted(glob.glob(os.path.join(input_dir,"sub*/func/sub*.nii.gz")))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    subjects = []
    for f in files:
        sub = f.split('/')[-1].split('_bld')[0]
        subjects.append(sub)
    unique_subjects = list(set(subjects))
    print(len(unique_subjects))

    subject_args = []
    for s in unique_subjects:
        runs = natsorted(glob.glob(os.path.join(input_dir,s+'*','func',s+'*')))
        subject_args.append((runs, s, mask_img, output_dir))

    # show the results for the first five subjects
    print(subject_args[:5])

    # make sure we got everyone
    number_of_subjects = len(subject_args)
    print(number_of_subjects)

    for i in tqdm(subject_args):
        transform(i)
