function [r, p] = matlab_perm_nmp(imgs1, imgs2, covar1, covar2, nperm, s1, s2)
%This function will take two sets of images (imgs1 and imgs2), compare them to two sets of 
%behaviors/covariates (covar1 and covar2) of sample size s1/s2, generate overall circuit maps,
%permute nperm times, and determine if the circuit maps are more similar than chance.
%imgs1 and imgs2 are M x N matrices in which N columns represent different patients, and N rows
%represent a column vector containing all of the voxels in each patient's seed map generated
%based on connectivity of the patient's lesion or stim site (or other ROI).
if size(covar1, 2) > 1
    nm1 = partialcorr(imgs1', covar1(:, 1), covar1(:, 2:end), 'type', 'pearson');
else
    nm1 = corr(imgs1', covar1, 'rows', 'complete');
end

if size(covar2, 2) > 1
    nm2 = partialcorr(imgs2', covar2(:, 1), covar2(:, 2:end), 'type', 'pearson');
else
    nm2 = corr(imgs2', covar2, 'rows', 'complete');
end


r = corr(nm1, nm2, 'rows','complete');


global parworkers
parfor_progress(nperm);

shuffle = zeros(nperm, 1);

parfor(i = 1:nperm, parworkers)
    rp1 = randperm(s1);
    rp2 = randperm(s2);
%control on dataset?
    if size(covar1, 2) > 1
        covar1_tmp1 = covar1(:, 1);
        covar1_tmp2 = covar1(:, 2:end);
        a = partialcorr(imgs1', covar1_tmp1(rp1), covar1_tmp2(rp1, :), 'type', 'pearson');
    else
        covar1_tmp = covar1;
        a = corr(imgs1', covar1_tmp, 'rows', 'complete');
    end
    if size(covar2, 2) > 1
        covar2_tmp1 = covar2(:, 1);
        covar2_tmp2 = covar2(:, 2:end);
        b = partialcorr(imgs2', covar2_tmp1(rp2), covar2_tmp2(rp2, :), 'type', 'pearson');
    else
        covar2_tmp = covar2;
        b = corr(imgs2', covar2_tmp, 'rows', 'complete');
    end

    shuffle(i) = corr(atanh(a), atanh(b), 'rows', 'complete');
    
    percent = parfor_progress;
    if mod(percent, 1) == 0
       disp(percent);
    end
end

parfor_progress(0); 

p = length(shuffle(abs(shuffle)>abs(r)));
p = p/(nperm + 1);



end