
# coding: utf-8

# In[ ]:

from __future__ import division, print_function

import numpy as np
from denspeed_std import nlmeans_3d_std


def nlmeans_std(arr, sigma, mean=None, mask=None, patch_radius=1, block_radius=5,
            rician=True, num_threads=None):
    """ Non-local means for denoising 3D and 4D images

    Parameters
    ----------
    arr : 3D or 4D ndarray
        The array to be denoised
    sigma : float or 3D array
        standard deviation of the noise estimated from the data
    mean : 3D ndarray
        if arr is standard deviation map (3D), mean is mean map of the same values
    mask : 3D ndarray
    patch_radius : int
        patch size is ``2 x patch_radius + 1``. Default is 1.
    block_radius : int
        block size is ``2 x block_radius + 1``. Default is 5.
    rician : boolean
        If True the noise is estimated as Rician, otherwise Gaussian noise
        is assumed.
    num_threads : int
        Number of threads. If None (default) then all available threads
        will be used (all CPU cores).

    Returns
    -------
    denoised_arr : ndarray
        the denoised ``arr`` which has the same shape as ``arr``.
    """

    if arr.ndim == 3:
        sigma = np.ones(arr.shape, dtype=np.float64) * sigma
        return nlmeans_3d_std(arr, mask, mean, sigma,
                          patch_radius, block_radius,
                          rician, num_threads).astype(arr.dtype)

    elif arr.ndim == 4:
        denoised_arr = np.zeros_like(arr)

        if isinstance(sigma, np.ndarray) and sigma.ndim == 3:
            sigma = (np.ones(arr.shape, dtype=np.float64) *
                     sigma[..., np.newaxis])
        else:
            sigma = np.ones(arr.shape, dtype=np.float64) * sigma

        for i in range(arr.shape[-1]):
            denoised_arr[..., i] = nlmeans_3d_std(arr[..., i],
                                              mask,
                                              None,
                                              sigma[..., i],
                                              patch_radius,
                                              block_radius,
                                              rician,
                                              num_threads).astype(arr.dtype)

        return denoised_arr

    else:
        raise ValueError("Only 3D or 4D array are supported!", arr.shape)

