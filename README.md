# Computing SNR for diffusion-weighted imaging
The following examples illustrate some basic SNR calculations on diffusion-weighted images (DWIs). The basic procedure is:

<ol>
<li>Correct for motion and for eddy current distortion, if necessary (i.e., if using single spin echo DWI; we use FSL's 'eddy')</li>

<li>For all the b=0 images, compute the standard deviation of each voxel. This will generate a noise map, but the map will be both biased and unreliable, due to the fact that there are only a few values used to estimate the stdev.</li>

<li>Correct the unreliability and bias by combining the stdev estimates across voxels. To define which voxels should be combined, use a patch based non-local means approach<sup>[1]</sup>. The non-local mean algorithms finds surrounding voxels with the same intensity.</li>

<li>For SNR of the non-DWIs (b=0 images), take the mean of all b0 images and divide by the noise estimate</li>

<li>For DWI SNR it's a little less clear what's best. A simple option is to use max across all the DWIs. That works OK for data with relatively high SNR, but could really over-estimate SNR for data where the true SNR is low. This is something I think we should discuss and try to come up with a better metric.</li>

<li>The thermal noise map is estimated with a PCA decomposition<sup>[2]</sup>. The part of the decomposition containing the least varying signal is used for the thermal noise.</li>
</ol>

<sup>[1]</sup> <i>[P. Coupe, P. Yger, S. Prima, P. Hellier, C. Kervrann, C. Barillot, “An Optimized Blockwise Non Local Means Denoising Filter for 3D Magnetic Resonance Images”, IEEE Transactions on Medical Imaging, 27(4):425-441, 2008]</i><br/>
<sup>[2]</sup> <i>[Diffusion Weighted Image Denoising Using Overcomplete Local PCA,  Manjon JV, Coupe P, Concha L, Buades A, Collins DL]</i><br/>
