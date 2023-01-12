# A Transdiagnostic Network for Psychiatric Illness Derived from Atrophy and Lesions. 

The code for our Nature Human Behavior paper *A Transdiagnostic Network for Psychiatric Illness Derived from Atrophy and Lesions* (https://doi.org/10.1038/s41562-022-01501-9).

## Environment installation
It is recommended to use [mamba](https://github.com/mamba-org/mamba) to install the conda environment:
```
mamba env create -f environment.yml
conda activate nhb_taylor2023
```
## Preparing the seeds
`sphere_maker.ipynb` can be used to generate spheres on a template brain from coordinates in a CSV file.

## Preparing the connectome

This code was written to be used with the [GSP1000 dataset](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/ILXIKS). The `format_connectome.py` script takes the 4D NiFTI files from the GSP1000 dataset, precomputes the algebraic norm per-voxel, and reshapes the data into 2D .npy files.

## Seed-based functional connectivity
`connectome_quick.py` accepts a text file where each line points to a different seed NiFTI and the directory of connectome files generated in the prior step. 
```
python connectome_quick.py \
    -r my_spheres.txt \
    -cs GSP1000_formatted \
    -o my_output
```

## Spatial permutation
Spatial permutation can be run with `spatial_permute.m`.

## References
- Dataset 1: https://doi.org/10.1001/jamapsychiatry.2014.2206
- Dataset 2: https://doi.org/10.1093/brain/awy292
- Dataset 4: https://doi.org/10.1038/npp.2010.132
