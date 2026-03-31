#! /usr/bin/env python
"""
Implementation of the STIM map from [PAI19]

.. [PAI19]
   | Pairet et al. 2019
   | **STIM map: detection map for exoplanets imaging beyond asymptotic Gaussian
   residual speckle noise**
   | *MNRAS, 487, 2262*
   | `http://doi.org/10.1093/mnras/stz1350
     <http://doi.org/10.1093/mnras/stz1350>`_

"""
__author__ = 'Benoit Pairet'
__all__ = ['stim_map',
           'inverse_stim_map',
           'normalized_stim_map',
           'make_stim2D_threshold',
           'create_distance_interpolated_array',
           'return_stim_max',
           'normalized_stim_pro']

import numpy as np
from ..preproc import cube_derotate
from ..var import get_circle, mask_circle
from scipy import signal


def gaussian_kernel(size: int, sigma: float):
    """Generates a 2D Gaussian kernel."""
    x_coord = np.arange(size) - size // 2
    x_grid, y_grid = np.meshgrid(x_coord, x_coord, indexing='ij')
    
    gaussian_kernel = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    #gaussian_kernel /= np.sum(gaussian_kernel)
    
    return gaussian_kernel



def masked_gaussian_convolution(image, mask, fwhm):
    """
    Applies a Gaussian convolution to the image only within the masked region.
    
    Parameters:
    image : 2D numpy array
        Input image to be convolved.
    mask : 2D boolean numpy array
        Mask indicating the region to convolve (True where convolution is applied).
    sigma : float
        Standard deviation of the Gaussian kernel.
        
    Returns:
    2D numpy array
        Convolved image with the same shape as the input, where only the masked region is convolved.
    """
    if mask is None:
        mask = np.ones_like(image)
    mask = np.array(mask, dtype = bool)
    
    # Create Gaussian kernel
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    kernel_size = 2 * int(3 * sigma) + 1


    kernel = gaussian_kernel(kernel_size, sigma)
    #plot_frames(kernel)
    
    # Compute numerator: convolution of (image * mask) with kernel
    numerator = signal.convolve2d(image * mask, kernel, mode='same', boundary='symm')
    
    # Compute denominator: convolution of mask with kernel
    denominator = signal.convolve2d(mask.astype(float), kernel, mode='same', boundary='symm')
    
    # Avoid division by zero by setting a small epsilon where denominator is zero
    eps = 1e-6
    denominator[denominator == 0] = eps
    
    # Compute the normalized convolution result for valid regions
    convolved_region = numerator / denominator
    
    # Apply the mask to retain only the convolved region
    result = image.copy()
    result[mask] = convolved_region[mask]
    result *= mask
    
    return result



def stim_map(cube_der):
    """Compute the STIM detection map as in [PAI19]_.

    Parameters
    ----------
    cube_der : 3d numpy ndarray
        Input de-rotated cube, e.g. ``residuals_cube_`` output from
        ``vip_hci.psfsub.pca``.

    Returns
    -------
    detection_map : 2d ndarray
        STIM detection map.

    """
    t, n, _ = cube_der.shape
    this_mean = np.mean(cube_der, axis=0)
    mu = this_mean#**2
    sigma = np.sqrt(np.var(cube_der, axis=0))
    detection_map = np.divide(mu, sigma, out=np.zeros_like(mu),
                              where=sigma != 0)
    return get_circle(detection_map, int(np.round(n/2.)))


def inverse_stim_map(cube, angle_list, **rot_options):
    """Compute the inverse STIM detection map as in [PAI19]_.

    Parameters
    ----------
    cube : 3d numpy ndarray
        Non de-rotated residuals from reduction algorithm, eg.
        ``residuals_cube`` output from ``vip_hci.psfsub.pca``.
    angle_list : 1d numpy ndarray
        Vector of derotation angles to align North up in your cube images.
    rot_options: dictionary, optional
        Dictionary with optional keyword values for "nproc", "imlib",
        "interpolation, "border_mode", "mask_val",  "edge_blend",
        "interp_zeros", "ker" (see documentation of
        ``vip_hci.preproc.frame_rotate``)

    Returns
    -------
    inv_stim_map : 2d ndarray
        Inverse STIM detection map.

    """
    t, n, _ = cube.shape
    cube_inv_der = cube_derotate(cube, -angle_list, **rot_options)
    inv_stim_map = stim_map(cube_inv_der)
    return inv_stim_map


def normalized_stim_map(cube, angle_list, mask=None, **rot_options):
    """Compute the normalized STIM detection map as in [PAI19]_.

    Parameters
    ----------
    cube : 3d numpy ndarray
        Non de-rotated residuals from reduction algorithm, eg.
        ``residuals_cube`` output from ``vip_hci.psfsub.pca``.
    angle_list : 1d numpy ndarray
        Vector of derotation angles to align North up in your cube images.
    mask : int, float, numpy ndarray 2d or None
        Mask informing where the maximum value in the inverse STIM map should
        be calculated. If an integer or float, a circular mask with that radius
        masking the central part of the image will be used. If a 2D array, it
        should be a binary mask (ones in the areas that can be used).
    rot_options: dictionary, optional
        Dictionary with optional keyword values for "nproc", "imlib",
        "interpolation, "border_mode", "mask_val",  "edge_blend",
        "interp_zeros", "ker" (see documentation of
        ``vip_hci.preproc.frame_rotate``)

    Returns
    -------
    normalized STIM map : 2d ndarray
        STIM detection map.

    """
    inv_map = inverse_stim_map(cube, angle_list, **rot_options)

    if mask is not None:
        if np.isscalar(mask):
            inv_map = mask_circle(inv_map, mask)
        else:
            inv_map *= mask

    max_inv = np.nanmax(inv_map)
    if max_inv <= 0:
        msg = "The normalization value is found to be {}".format(max_inv)
        raise ValueError(msg)

    cube_der = cube_derotate(cube, angle_list, **rot_options)
    det_map = stim_map(cube_der)

    return det_map/max_inv


def create_distance_interpolated_array(values, shape):
    """
    Create a 2D array where each pixel's value is determined by its distance to the center,
    using linear interpolation from the given values vector.

    Parameters:
    values (list or np.ndarray): The 1D vector of values to interpolate from.
    shape: (height, width) of output 2D array

    Returns:
        np.ndarray: The resulting 2D array with interpolated values based on distance from the center.
    """
    values = np.asarray(values)
    if len(values) == 0:
        raise ValueError("Values vector must not be empty.")

    height = shape[0]
    width = shape[1]

    # Calculate center coordinates
    y_center = height // 2
    x_center = width // 2

    # Generate grid of indices
    y_indices, x_indices = np.indices((height, width))

    # Compute =distance from the center for each pixel
    distances = np.sqrt((x_indices - x_center)**2 + (y_indices - y_center)**2)

    # Compute floor and ceiling indices for each distance
    floor_d = np.floor(distances).astype(int)
    ceil_d = floor_d + 1
    alpha = distances - floor_d  # Fractional part for interpolation

    # Clamp indices to valid range [0, len(values)-1]
    max_index = len(values) - 1
    floor_d = np.clip(floor_d, 0, max_index)
    ceil_d = np.clip(ceil_d, 0, max_index)

    # Perform linear interpolation
    interpolated = (1 - alpha) * values[floor_d] + alpha * values[ceil_d]
    
    #safeguard against potential zeros or negative values the might have creeped in somehow
    max_value = np.nanmax(interpolated)
    interpolated[np.where(interpolated <= 0)] = max_value

    return interpolated


def return_stim_max(stim, fwhm = 4, width = 1, mask = None):
    """
    Create a 1D array where each value corresponds to the maximum of the stim
    array in an annulus of width given by width*fwhm in pixels and centered 
    on increasing distances to the center of the stim array

    Parameters:
    stim : The 2D array of (inverse) stim values
    fwhm: fwhm for the observation
    width: width (in fwhm) of each annulus
    mask: binary mask that may exclude some areas of the image from being taken
        into account for the maxima computation

    Returns:
        np.ndarray: 1D vector containing each maximum starting at distance
            equal to one
    """
    y,x = stim.shape

    values = np.zeros(int(x/2))
    means = np.zeros(int(x/2))
    stds = np.zeros(int(x/2))
    
    #To be compatible with with arrays of fwhm (for 4D datasets for example)
    if not np.isscalar(fwhm):
        fwhm = np.mean(fwhm)

    factor = 2 / width
    for r in range(int(x/2)):
        this_mask = np.ones((y,x))
        this_min = np.max((0,r-fwhm/factor))
        this_max = np.min((r+fwhm/factor, x/2))
        this_mask = mask_circle(this_mask, this_min)
        this_mask = mask_circle(this_mask, this_max, mode = 'out')
        if mask is not None:
            this_mask *= mask
            
        yy,xx = np.where(this_mask == 1)

        values[r] = np.nanmax(stim*this_mask)
        means[r] = np.nanmean(stim[yy,xx])
        stds[r] = np.nanstd(stim[yy,xx])

    values[np.where(values <= 0)] = np.nanmax(values)
    return values, means, stds


def make_stim2D_threshold(inv_stim, fwhm = 4, width = 1, mask = None):
    """
    Returns map of stim 2D threshold depending on the distance to the center of 
    the frame
    
    Parameters:
    inv_stim : The 2D array of (inverse) stim values
    fwhm: fwhm for the observation
    width: width (in fwhm) of each annulus
    mask: binary mask that may exclude some areas of the image from being taken
        into account for the maxima computation
    """
    
    values, means, stds = return_stim_max(inv_stim, mask = mask, fwhm = fwhm, width = width)
    
    result = create_distance_interpolated_array(values, inv_stim.shape)
    means = create_distance_interpolated_array(means, inv_stim.shape)
    stds = create_distance_interpolated_array(stds, inv_stim.shape)
    
    #if mask is not None:
    #    result *= mask
    
    return result, means, stds


def normalized_stim_pro(res_, res, derot = None, conv = True, fwhm = 4, width = 1.5, mask= None):
    
    """
    res_ is the derotated residuals.
    res is the inversely derotated residuals
    
    returns the progessively thresholded stim map, in units of standard deviation
    above the mean of the inverse stim map
    
    if conv is True, both the direct and the inverse stim map are convolved before
    the thresholding
    
    if derot is None, no additional derotation is done
    fs it is not None, it should be the derotation angle of the cube, whose inverse
    will be used to derotate res and NOT res_ which should already be derotated

    """

    
    
    if derot is not None:
        res = cube_derotate(res, -derot, imlib = 'opencv', interpolation = 'lanczos4')
    
    if mask is None:
        mask = np.ones_like(res[0])
        
        
    stim = stim_map(res_)
    inv = stim_map(res)
        
    if conv:
        inv = masked_gaussian_convolution(inv, mask, fwhm)
        #print('Max with convolution: {}'.format(np.nanmax(inv)))
        stim = masked_gaussian_convolution(stim, mask, fwhm)
        
    
    this_max, this_mean, this_std = make_stim2D_threshold(inv, fwhm, width, mask)
    
    stim = (stim - this_mean) / this_std
    
    return stim
        
        
        
        
