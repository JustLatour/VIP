#! /usr/bin/env python
"""
Module with completeness curve and map generation function.

.. [DAH21b]
   | Dahlqvist et al. 2021b
   | **Auto-RSM: An automated parameter-selection algorithm for the RSM map
     exoplanet detection algorithm**
   | *Astronomy & Astrophysics, Volume 656, Issue 2, p. 54*
   | `https://arxiv.org/abs/2109.14318
     <https://arxiv.org/abs/2109.14318>`_

.. [JEN18]
   | Jensen-Clem et al. 2018
   | **A New Standard for Assessing the Performance of High Contrast Imaging
     Systems**
   | *The Astrophysical Journal, Volume 155, Issue 1, p. 19*
   | `https://arxiv.org/abs/1711.01215
     <https://arxiv.org/abs/1711.01215>`_

.. [MAW14]
   | Mawet et al. 2014
   | **Fundamental Limitations of High Contrast Imaging Set by Small Sample
     Statistics**
   | *The Astrophysical Journal, Volume 792, Issue 1, p. 97*
   | `https://arxiv.org/abs/1407.2247
     <https://arxiv.org/abs/1407.2247>`_

"""

__author__ = "C.H. Dahlqvist, V. Christiaens, T. Bédrine"
__all__ = ["completeness_curve", "completeness_map", "completeness_curve_stim", 
           "plot_fc_results", "completeness_theta_map", "completeness_curve_snr"]

from math import gcd
from inspect import getfullargspec
from multiprocessing import cpu_count

import numpy as np
from astropy.convolution import convolve, Tophat2DKernel
from matplotlib import pyplot as plt
from skimage.draw import disk

from .contrcurve import contrast_curve
from .snr_source import snrmap, _snr_approx, snr, significance
from .stim import normalized_stim_map, stim_map, inverse_stim_map, make_stim2D_threshold
from ..config.utils_conf import pool_map, iterable, vip_figsize, vip_figdpi
from ..fm import cube_inject_companions, normalize_psf
from ..fm.utils_negfc import find_nearest
from ..preproc import cube_crop_frames
from ..var import get_annulus_segments, frame_center, mask_circle, frame_filter_lowpass

from hciplot import plot_frames
from photutils.aperture import CircularAperture, aperture_photometry
from ..config.paramenum import (
    SvdMode,
    Adimsdi,
    Interpolation,
    Imlib,
    Collapse,
    ALGO_KEY,
)



from scipy import signal
from scipy.optimize import curve_fit


def gaussian(x, A, mu, sigma):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))


def plot_Zmap(image, fwhm, mode = 'stim', minr = 1, maxr = 3, inv_image = None, exclude_negative_lobes = False, plot = True,
             nbr_bins = 24, mask = None):

    minr *= fwhm
    maxr *= fwhm

    if mask is None:
        mask = np.ones_like(image)
    mask = mask_circle(mask, minr)
    mask = mask_circle(mask, maxr, mode = 'out')

    yy,xx = np.where(mask == 1)

    bin_edges = nbr_bins
    
    if mode == 'stim':

        inv_imageC = frame_filter_lowpass(inv_image, fwhm_size = fwhm)
        imageC = frame_filter_lowpass(image, fwhm_size = fwhm)

        dataDC = imageC[yy,xx]
        dataIC = inv_imageC[yy,xx]

        countsI, bin_edges = np.histogram(dataIC, bins='fd')

        space = bin_edges[1] - bin_edges[0]
        minV = min((np.min(dataDC), np.min(dataIC)))
        maxV = max((np.max(dataDC), np.max(dataIC)))
        full_bin_edges = list(bin_edges)

        while full_bin_edges[-1] < maxV:
            full_bin_edges.append(full_bin_edges[-1] + space)

        while full_bin_edges[0] < minV:
            full_bin_edges.insert(0, full_bin_edges[0] - space)

        full_bin_edges = np.array(full_bin_edges)
        
        if plot:
            countsD, bin_edges, _ = plt.hist(dataDC, bins=full_bin_edges)
            countsI, bin_edges, _ = plt.hist(dataIC, bins=full_bin_edges, alpha = 0.5)
        else:
            countsD, bin_edges = np.histogram(dataDC, bins=full_bin_edges)
            countsI, bin_edges = np.histogram(dataIC, bins=full_bin_edges)

        #counts, bin_edges, p = plt.hist(dataIC, bins = bin_edges)
        
        bin_centers = 0.5 * (full_bin_edges[:-1] + full_bin_edges[1:])

        p0 = [max(countsI), np.mean(dataIC), np.std(dataIC)]   # initial guesses
        params, cov = curve_fit(gaussian, bin_centers, countsI, p0=p0)

        A_fit, mu_fit, sigma_fit = params

        this_mu = np.mean(dataIC)
        this_sigma = np.std(dataIC)

        if plot:
            x = np.linspace(full_bin_edges[0], full_bin_edges[-1], 400)
            plt.plot(x, gaussian(x, *params), '-', label='Gaussian fit', color = 'red')
            plt.show()

        Zmap = (imageC - this_mu)/this_sigma
        
        return Zmap, mu_fit, sigma_fit
        
    elif mode == 'sig':
        Zmap = snrmap(image, fwhm, exclude_negative_lobes = exclude_negative_lobes, plot = plot)

        return Zmap


def get_adi_res(cube, collapse_ifs = 'mean'):
    cube = np.array(cube)
    nc, nz, ny, nx = cube.shape

    if collapse_ifs == 'mean':
        result = np.mean(cube[:,:,:,:], axis = 0)
    elif collapse_ifs == 'median':
        result = np.median(cube[:,:,:,:], axis = 0)
    else:
        raise TypeError('Collapse mode not recognized')

    return result

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
    
    def gaussian_kernel(size: int, sigma: float):
        """Generates a 2D Gaussian kernel."""
        x_coord = np.arange(size) - size // 2
        x_grid, y_grid = np.meshgrid(x_coord, x_coord, indexing='ij')
    
        gaussian_kernel = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    
        return gaussian_kernel
    
    if mask is None:
        mask = np.ones_like(image)
    mask = np.array(mask, dtype = bool)
    
    if not np.isscalar(fwhm):
        fwhm = np.mean(fwhm)
    
    # Create Gaussian kernel
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    kernel_size = 2 * int(3 * sigma) + 1


    kernel = gaussian_kernel(kernel_size, sigma)
    
    # Compute numerator: convolution of (image * mask) with kernel
    numerator = signal.convolve2d(np.array(image * mask), np.array(kernel), mode='same', boundary='symm')
    
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


def _estimate_snr_fc(
     a,
     an_dist,
     b,
     level,
     n_fc,
     cube,
     psf,
     angle_list,
     fwhm,
     algo,
     algo_dict,
     through_thresh,
     sigma=5,
     exclude_negative_lobes=True,
     starphot=1,
     approximated=False,
     transmission=None,
     check_max=False
):
    if np.isscalar(starphot):
        flevel = level * starphot
    elif isinstance(starphot, list):
        starphot = np.array(starphot)
    if isinstance(starphot, np.ndarray):
        flevel = level * np.mean(starphot)
    
    radial_gradient = False
    if transmission is not None:
        radial_gradient = True
    
    if isinstance(cube, np.ndarray):
        cubefc = cube_inject_companions(
            cube,
            psf,
            angle_list,
            flevel=flevel,
            rad_dists=a,
            theta=b / n_fc * 360,
            n_branches=1,
            verbose=False,
            transmission=transmission,
            radial_gradient=radial_gradient
            )
    else:
        cubefc = []
        for i in range(len(cube)):
            this_cubefc = cube_inject_companions(
                cube[i],
                psf[i],
                angle_list[i],
                flevel=flevel,
                rad_dists=a,
                theta=b / n_fc * 360,
                n_branches=1,
                verbose=False,
                transmission=transmission,
                radial_gradient=radial_gradient
                )
            cubefc.append(this_cubefc)

    if isinstance(fwhm, (np.ndarray, list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if cube.ndim == 4:
        cy, cx = frame_center(cube[0, 0, :, :])
    else:
        cy, cx = frame_center(cube[0])

    # TODO: Clean below?
    # Consider 3 cases depending on whether algo is (i) defined externally,
    # (ii) a VIP postproc algorithm; (iii) ineligible for contrast curves
    argl = getfullargspec(algo).args
    if "cube" in argl and "angle_list" in argl and "verbose" in argl:
        # (i) external algorithm with appropriate parameters [OK]
        pass
    else:
        algo_name = algo.__name__
        idx = algo.__module__.index(
            '.', algo.__module__.index('.') + 1)
        mod = algo.__module__[:idx]
        tmp = __import__(
            mod, fromlist=[algo_name.upper()+'_Params'])
        algo_params = getattr(tmp, algo_name.upper()+'_Params')
        argl = [attr for attr in vars(algo_params)]
        if "cube" in argl and "angle_list" in argl and "verbose" in argl:
            # (ii) a VIP postproc algorithm [OK]
            pass
        else:
            # (iii) ineligible routine for contrast curves [Raise error]
            msg = "Ineligible algo for contrast curve function. algo should "
            msg += "have parameters 'cube', 'angle_list' and 'verbose'"
            raise TypeError(msg)
    # algo_params = signature(algo).parameters
    # param_name = next(iter(algo_params))
    # class_name = algo_params[param_name].annotation

    # TODO: Clean below?
    # Consider 3 cases depending on whether algo is (i) defined externally,
    # (ii) a VIP postproc algorithm; (iii) ineligible for contrast curves
    argl = getfullargspec(algo).args
    algo_name = algo.__name__
    if "cube" in argl and "angle_list" in argl and "verbose" in argl:
        # (i) external algorithm with appropriate parameters [OK]
        pass
    else:
        idx = algo.__module__.index(
            '.', algo.__module__.index('.') + 1)
        mod = algo.__module__[:idx]
        tmp = __import__(
            mod, fromlist=[algo_name.upper()+'_Params'])
        algo_params = getattr(tmp, algo_name.upper()+'_Params')
        argl = [attr for attr in vars(algo_params)]
        if "cube" in argl and "angle_list" in argl and "verbose" in argl:
            # (ii) a VIP postproc algorithm [OK]
            pass
        else:
            # (iii) ineligible routine for contrast curves [Raise error]
            msg = "Ineligible algo for contrast curve function. algo should "
            msg += "have parameters 'cube', 'angle_list' and 'verbose'"
            raise TypeError(msg)
    # algo_params = signature(algo).parameters
    # param_name = next(iter(algo_params))
    # class_name = algo_params[param_name].annotation

    # argl = [attr for attr in vars(class_name)]
    #if "verbose" in argl:
    #    algo_dict["verbose"] = False
    if "fwhm" in argl:
        algo_dict["fwhm"] = fwhm_med
    if "annular" in algo_name:
        output_temp = algo(
            cube=cubefc, angle_list=angle_list,
            full_output = True,
            **algo_dict
        )
        frame_fin = output_temp[2]
        residuals_ = output_temp[1]
        residuals = output_temp[0]
    elif algo_name == 'pca':
        output_temp = algo(cube=cubefc, angle_list=angle_list, 
                         full_output = True, **algo_dict)
        
        
        if len(cubefc.shape) == 4:
            if algo_dict['scale_list'] is None:
                frame_fin = output_temp[0]

            else:
                if (algo_dict['adimsdi'] == Adimsdi.DOUBLE or 
                                   algo_dict['cube_ref'] is not None):
                    frame_fin = output_temp[0]
                else:
                    frame_fin = output_temp[0]
                
        else:
            frame_fin = output_temp[0]

    elif '4S' in algo_name or 'FourS' in algo_name:
        output_temp = algo(cube=cubefc, angle_list=angle_list, 
                         **algo_dict)
        
        frame_fin = output_temp[2]

    if 'pca' in algo_name:
        ncomp = algo_dict['ncomp']
        if np.isscalar(ncomp):
            ncomp = np.array([ncomp])
        else:
            ncomp = np.array(ncomp)
        nncomp = len(ncomp)
        result = np.zeros((nncomp))
    else:
        nncomp = 1
        result = np.zeros(1)
        ncomp = [0]
        
    
    mask = np.ones_like(frame_fin)
    bmask = np.ma.make_mask(mask)
    yy, xx = np.where(bmask)
    

    y, x = frame_fin.shape
    twopi = 2 * np.pi
    sigposy = int(y / 2 + np.sin(b / n_fc * twopi) * a)
    sigposx = int(x / 2 + np.cos(b / n_fc * twopi) * a)
    
    indc = disk((sigposy, sigposx), fwhm_med/1.5)
    indc1 = disk((sigposy, sigposx), fwhm_med/2)
    
    this_a = np.where(np.array(an_dist) == a)[0][0]

    annulus = np.ones_like(frame_fin[0])
    if check_max:
        annulus = mask_circle(annulus, a-fwhm_med)
        annulus = mask_circle(annulus, a+fwhm_med, mode = 'out')
        
    snrmaps = np.zeros((nncomp, frame_fin.shape[-2], frame_fin.shape[-1]))
    
    for i,n in enumerate(ncomp):
        
        snrmaps[i] = snrmap(frame_fin, fwhm = fwhm, exclude_negative_lobes = exclude_negative_lobes, 
               approximated = approximated, verbose = False)
        
        
        pxl_values = np.nan_to_num(snrmaps[i][indc[0], indc[1]])
        these_indices = np.where(pxl_values>0)
        
        
        pxl_values = np.nan_to_num(snrmaps[i][indc[0], indc[1]])

        if len(these_indices[0]) <= 1:
            result[i] = 0
        else:
            this_v = np.nanmax(pxl_values[these_indices])
            this_v = significance(this_v, a, fwhm = fwhm_med, verbose = False)
                
            #result[i] = this_v - stim_thresh[i][3]
            result[i] = this_v - sigma
            
        if check_max:
            max_map = np.nan_to_num(snrmaps[i]*annulus).max()
            max_target = np.nan_to_num(snrmaps[i][indc[0], indc[1]]).max()
            if max_map > max_target:
                result[i] = 0
            
        
    if nncomp == 1:
        result = result[0]

    return result, b, snrmaps, frame_fin


def _stim_fc(
    a,
    an_dist,
    b,
    level,
    n_fc,
    cube,
    psf,
    angle_list,
    fwhm,
    algo,
    algo_dict,
    stim_thresh,
    through_thresh=0,
    mask=None,
    conv=False,
    starphot=1,
    transmission=None,
    snr=False,
    check_max=False,
    inv_stim=None,
    sigma=5,
    Zthresh = None,
    width=1.5
):
    
    if np.isscalar(starphot):
        flevel = level * starphot
    elif isinstance(starphot, list):
        starphot = np.array(starphot)
    if isinstance(starphot, np.ndarray):
        flevel = level * np.mean(starphot)
    
    radial_gradient = False
    if transmission is not None:
        radial_gradient = True
        
    if isinstance(cube, np.ndarray):
        cubefc = cube_inject_companions(
            cube,
            psf,
            angle_list,
            flevel=flevel,
            rad_dists=a,
            theta=b / n_fc * 360,
            n_branches=1,
            verbose=False,
            transmission=transmission,
            radial_gradient=radial_gradient
            )
    else:
        cubefc = []
        for i in range(len(cube)):
            this_cubefc = cube_inject_companions(
                cube[i],
                psf[i],
                angle_list[i],
                flevel=flevel,
                rad_dists=a,
                theta=b / n_fc * 360,
                n_branches=1,
                verbose=False,
                transmission=transmission,
                radial_gradient=radial_gradient
                )
            cubefc.append(this_cubefc)

    if isinstance(fwhm, (np.ndarray, list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if isinstance(cube, list):
        cy, cx = frame_center(cube[0][0])
    elif cube.ndim == 4:
        cy, cx = frame_center(cube[0, 0, :, :])
    else:
        cy, cx = frame_center(cube[0])

    # TODO: Clean below?
    # Consider 3 cases depending on whether algo is (i) defined externally,
    # (ii) a VIP postproc algorithm; (iii) ineligible for contrast curves
    argl = getfullargspec(algo).args
    algo_name = algo.__name__
    if "cube" in argl and "angle_list" in argl and "verbose" in argl:
        # (i) external algorithm with appropriate parameters [OK]
        pass
    else:
        idx = algo.__module__.index(
            '.', algo.__module__.index('.') + 1)
        mod = algo.__module__[:idx]
        tmp = __import__(
            mod, fromlist=[algo_name.upper()+'_Params'])
        algo_params = getattr(tmp, algo_name.upper()+'_Params')
        argl = [attr for attr in vars(algo_params)]
        if "cube" in argl and "angle_list" in argl and "verbose" in argl:
            # (ii) a VIP postproc algorithm [OK]
            pass
        else:
            # (iii) ineligible routine for contrast curves [Raise error]
            msg = "Ineligible algo for contrast curve function. algo should "
            msg += "have parameters 'cube', 'angle_list' and 'verbose'"
            raise TypeError(msg)
    # algo_params = signature(algo).parameters
    # param_name = next(iter(algo_params))
    # class_name = algo_params[param_name].annotation

    # argl = [attr for attr in vars(class_name)]
    #if "verbose" in argl:
    #    algo_dict["verbose"] = False
    if "fwhm" in argl:
        algo_dict["fwhm"] = fwhm_med
    if "annular" in algo_name:
        output_temp = algo(
            cube=cubefc, angle_list=angle_list,
            full_output = True,
            **algo_dict
        )
        frame_fin = output_temp[2]
        residuals_ = output_temp[1]
        residuals = output_temp[0]
    elif algo_name == 'pca':
        output_temp = algo(cube=cubefc, angle_list=angle_list, 
                         full_output = True, **algo_dict)
        
        
        if len(cubefc.shape) == 4:
            if algo_dict['scale_list'] is None:
                frame_fin = output_temp[0]
                residuals = output_temp[3]
                residuals_ = output_temp[4]
            else:
                if (algo_dict['adimsdi'] == Adimsdi.DOUBLE or 
                                   algo_dict['cube_ref'] is not None):
                    frame_fin = output_temp[0]
                    residuals = output_temp[1]
                    residuals_ = output_temp[2]
                else:
                    frame_fin = output_temp[0]
                    residuals = output_temp[2]
                    residuals_ = output_temp[3]
                
            to_collapse = False
            if algo_dict['cube_ref'] is not None:
                to_collapse = True
            if algo_dict['scale_list'] is None:
                to_collapse = True
                
            if to_collapse:
                residuals_ = get_adi_res(residuals_)
                residuals = get_adi_res(residuals)
        else:
            frame_fin = output_temp[0]
            residuals_ = output_temp[4]
            residuals = output_temp[3]
    elif '4S' in algo_name or 'FourS' in algo_name:
        output_temp = algo(cube=cubefc, angle_list=angle_list, 
                         **algo_dict)
        
        frame_fin = output_temp[2]
        residuals_ = output_temp[1]
        residuals = output_temp[0]

    if 'pca' in algo_name:
        ncomp = algo_dict['ncomp']
        if np.isscalar(ncomp):
            ncomp = np.array([ncomp])
        else:
            ncomp = np.array(ncomp)
        nncomp = len(ncomp)
        result = np.zeros((nncomp))
    else:
        nncomp = 1
        result = np.zeros(1)
        ncomp = [0]
    
    stim_map_fc = np.zeros((nncomp, frame_fin.shape[-2], frame_fin.shape[-1]))
    cy, cx = frame_center(frame_fin)
    
    y, x = frame_fin.shape
    twopi = 2 * np.pi
    sigposy = int(y / 2 + np.sin(b / n_fc * twopi) * a)
    sigposx = int(x / 2 + np.cos(b / n_fc * twopi) * a)
    
    indc = disk((sigposy, sigposx), fwhm_med/1.5)
    indc1 = disk((sigposy, sigposx), fwhm_med/2)
    
    this_a = np.where(np.array(an_dist) == a)[0][0]
    
    if nncomp == 1:
        residuals = residuals.reshape(1,residuals.shape[0], 
                                    residuals.shape[1],residuals.shape[2])
        residuals_ = residuals_.reshape(1,residuals_.shape[0], 
                                    residuals_.shape[1],residuals_.shape[2])
        frame_fin = frame_fin.reshape(1, frame_fin.shape[0], frame_fin.shape[1])
        
    if snr:
        snr_value = np.zeros((nncomp,2))
        stim_value = np.zeros(nncomp)
        throughput = np.zeros(nncomp)
        snrmaps = np.zeros((nncomp, frame_fin.shape[-2], frame_fin.shape[-1]))
        
        
    annulus = np.ones_like(frame_fin[0])
    if check_max:
        annulus = mask_circle(annulus, a-fwhm_med)
        annulus = mask_circle(annulus, a+fwhm_med, mode = 'out')
    
    for i,n in enumerate(ncomp):
        
        stim_map_fc[i] = stim_map(residuals_[i])
        
        if Zthresh is not None:
            Zmap,_,_ = plot_Zmap(stim_map_fc[i], fwhm_med, mode = 'stim', mask = mask,
                       minr = (a/fwhm_med)-width/2, maxr = (a/fwhm_med)+width/2, inv_image = inv_stim,
                      exclude_negative_lobes = False, plot = False, nbr_bins = 24)
            
            pxl_values = np.nan_to_num(Zmap[indc[0], indc[1]])
            
            result = np.nanmax(pxl_values) - Zthresh
            
            return result, b, stim_map_fc, Zmap, frame_fin
        
        else:
            stim_map_fc[i] /= stim_thresh[i][0]
        
            if conv:
                stim_map_fc[i] = masked_gaussian_convolution(stim_map_fc[i], mask, fwhm)
        
        #if mask is not None:
        #    stim_map_fc[i] *= mask
        
        #max_target = np.nan_to_num(snrmap_fin[indc[0], indc[1]]).max()
        #mean_target = np.nan_to_num(stim_map_fc[i][indc[0], indc[1]]).mean()
        #stim_map_fc[i][indc[0], indc[1]] = 0
        
            pxl_values = np.nan_to_num(stim_map_fc[i][indc[0], indc[1]])
            these_indices = np.where(pxl_values>0)
            
        
            stim_map_fc[i] -= stim_thresh[i][3]
            pxl_values = np.nan_to_num(stim_map_fc[i][indc[0], indc[1]])
        
        
            if len(these_indices[0]) <= 1:
                result[i] = 0
            else:
                if conv:
                    this_v = np.nanmax(pxl_values[these_indices])
                else:
                    this_v = np.mean(pxl_values[these_indices])
                
                #result[i] = this_v - stim_thresh[i][3]
                result[i] = this_v
            
            apertures = CircularAperture((sigposx, sigposy), fwhm_med / 2)
            this_flux = aperture_photometry(frame_fin[i], apertures)
            this_flux = np.array(this_flux["aperture_sum"])[0]
            recovered_flux = this_flux - stim_thresh[i][4][this_a,b]
            this_throughput = recovered_flux/flevel
        
            if this_throughput < through_thresh:
                result[i] = 0
            
            if check_max:
                max_map = np.nan_to_num(stim_map_fc[i]*annulus).max()
                max_target = np.nan_to_num(stim_map_fc[i][indc[0], indc[1]]).max()
                if max_map > max_target:
                    result[i] = 0
            
            if snr:
                if len(these_indices[0]) <= 1:
                    stim_value[i] = 0
                else:
                    stim_value[i] = this_v
                throughput[i] = this_throughput
                snrmaps[i] = snrmap(frame_fin[i], fwhm, approximated = True, plot = False, verbose = False)
                snrpxlvalues = np.nan_to_num(snrmaps[i][indc1[0], indc1[1]])
                these_indices = np.where(snrpxlvalues>0)[0]
                if len(these_indices) > 0:
                    snr_value[i,0] = np.nanmax(snrpxlvalues[these_indices])
                    snr_value[i,1] = np.nanmean(snrpxlvalues[these_indices])
                else:
                    snr_value[i,0] = np.nanmax(snrpxlvalues)
                    snr_value[i,1] = np.nanmean(snrpxlvalues)
            

        #result[i] = max_target-max_map
        #result[i] = mean_target - 1
        
        #if mean_target1 < 1:
        #    result[i] = -1
        
        #if mean_target1 > mean_target0:
        #    result[i] = -1
        #if mean_target2 > mean_target1:
        #    result[i] = -1
        #if mean_target2 > mean_target0:
        #    result[i] = -1
            
        #print(mean_target0, mean_target1, mean_target2)
        
    if nncomp == 1:
        result = result[0]

    if snr:
        return result, b, stim_map_fc, frame_fin, stim_value, snr_value, snrmaps, throughput
    else:
        return result, b, stim_map_fc, frame_fin


# TODO: Include algo_class modifications in any tutorial using this function
def completeness_curve(
    cube,
    angle_list,
    psf,
    fwhm,
    algo,
    an_dist=None,
    ini_contrast=None,
    starphot=1,
    pxscale=0.1,
    n_fc=20,
    completeness=0.95,
    snr_approximation=True,
    max_iter=50,
    nproc=1,
    algo_dict={},
    verbose=True,
    plot=True,
    dpi=vip_figdpi,
    save_plot=None,
    object_name=None,
    fix_y_lim=(),
    figsize=vip_figsize,
    algo_class=None,
):
    """
    Function allowing the computation of completeness-based contrast curves with
    any of the psf-subtraction algorithms provided by VIP. The code relies on
    the approach proposed in [DAH21b]_, itself inspired by the framework
    developed in [JEN18]_. It relies on the computation of the contrast
    associated to a completeness level achieved at a level defined as the first
    false positive in the original SNR map (brightest speckle observed in the
    empty map) instead of the computation o the local noise and throughput (see
    the ``vip_hci.metrics.contrast_curve`` function). The computation of the
    completeness level associated to a contrast is done via the sequential
    injection of multiple fake companions. The algorithm uses multiple
    interpolations to find the contrast associated to the selected completeness
    level (0.95 by default). More information about the algorithm can be found
    in [DAH21b]_.

    Parameters
    ----------
    cube : 3d or 4d numpy ndarray
        The input cube, 3d (ADI data) or 4d array (IFS data), without fake
        companions.
    angle_list : 1d numpy ndarray
        Vector of derotation angles to align North up in your cube images.
    psf : 2d or 3d numpy ndarray
        Frame with the psf template for the fake companion(s).
        PSF must be centered in array. Normalization is done internally.
    fwhm: int or float or 1d array, optional
        The the Full Width Half Maximum in pixels. It can handle a different
        FWHM value for different wavelengths (IFS data).
    algo : callable or function
        The post-processing algorithm, e.g. ``vip_hci.pca.pca``.
    an_dist: list or ndarray, optional
        List of angular separations for which a contrast has to be estimated.
        Default is None which corresponds to a range spanning 2 FWHM to half
        the size of the provided cube - PSF size //2 with a step of 5 pixels
    ini_contrast: list, 1d ndarray or None, optional
        Initial contrast for the range of angular separations included in
        `an_dist`. The number of initial contrasts should be equivalent to the
        number of angular separations. Default is None which corresponds to the
        5-sigma contrast_curve obtained with ``vip_hci.metrics.contrast_curve``.
    starphot : int or float or 1d array, optional
        If int or float it corresponds to the aperture photometry of the
        non-coronagraphic PSF which we use to scale the contrast.
        Default is 1, which corresponds to an output contrast expressed in ADU.
    pxscale : float, optional
        Plate scale or pixel scale of the instrument. Only used for plots.
    n_fc: int, optional
        Number of azimuths considered for the computation of the True
        positive rate/completeness,(number of fake companions injected
        sequentially). The number of azimuths is defined such that the
        selected completeness is reachable (e.g. 95% of completeness
        requires at least 20 fake companion injections). Default 20.
    completeness: float, optional
        The completeness level to be achieved when computing the contrasts,
        i.e. the True positive rate reached at the threshold associated to
        the first false positive (the first false positive is defined as
        the brightest speckle present in the entire detection map).
        Default 95.
    snr_approximation : bool, optional
        If True, an approximated S/N map is generated. If False the
        approach of [MAW14]_ is used. Default is True, for speed.
    max_iter: int, optional
        Maximum number of iterations to consider in the search for the contrast
        level achieving desired completeness before considering it unachievable.
    nproc : int or None, optional
        Number of processes for parallel computing.
    algo_dict: dictionary, optional
        Any other valid parameter of the post-processing algorithms can be
        passed here, including e.g. imlib and interpolation.
    verbose : bool, optional
        Whether to print more info while running the algorithm. Default: True.
    plot : bool, optional
        Whether to plot the final contrast curve or not. True by default.
    dpi : int optional
        Dots per inch for the plots. 100 by default. 300 for printing quality.
    save_plot: string or None, optional
        If provided, the contrast curve will be saved to this path.
    object_name: string or None, optional
        Target name, used in the plot title.
    fix_y_lim: tuple, optional
        If provided, the y axis limits will be fixed, for easier comparison
        between plots.
    fig_size: tuple, optional
        Figure size

    Returns
    -------
    an_dist: 1d numpy ndarray
        Radial distances where the contrasts are calculated
    cont_curve: 1d numpy ndarray
        Contrasts for the considered radial distances and selected completeness
        level.
    """

    if (100 * completeness) % (100 / n_fc) > 0:
        n_fc = int(100 / gcd(int(100 * completeness), 100))

    if cube.ndim != 3 and cube.ndim != 4:
        raise TypeError("The input array is not a 3d or 4d cube")
    if cube.ndim == 3 and (cube.shape[0] != angle_list.shape[0]):
        raise TypeError("Input parallactic angles vector has wrong length")
    if cube.ndim == 4 and (cube.shape[1] != angle_list.shape[0]):
        raise TypeError("Input parallactic angles vector has wrong length")
    if cube.ndim == 3 and psf.ndim != 2:
        raise TypeError("Template PSF is not a frame (for ADI case)")
    if cube.ndim == 4 and psf.ndim != 3:
        raise TypeError("Template PSF is not a cube (for ADI+IFS case)")
    if nproc is None:
        nproc = cpu_count() // 2

    if isinstance(fwhm, (np.ndarray, list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if an_dist is None:
        an_dist = np.array(
            range(2 * round(fwhm_med),
                  int(cube.shape[-1] // 2 - 2 * fwhm_med), 5)
        )
        print("an_dist not provided, the following list will be used:", an_dist)
    elif an_dist[-1] > cube.shape[-1] // 2 - 2 * fwhm_med:
        raise TypeError("Please decrease the maximum annular distance")

    if ini_contrast is None:
        print("Contrast curve not provided => will be computed first...")
        ini_cc = contrast_curve(
            cube,
            angle_list,
            psf,
            fwhm_med,
            pxscale,
            starphot,
            algo,
            sigma=3,
            nbranch=1,
            theta=0,
            inner_rad=1,
            wedge=(0, 360),
            fc_snr=100,
            plot=False,
            algo_class=algo_class,
            **algo_dict,
        )
        ini_rads = np.array(ini_cc["distance"])
        ini_cc = np.array(ini_cc["sensitivity_student"])

        if np.amax(an_dist) > np.amax(ini_rads):
            msg = "Max requested annular distance larger than covered by "
            msg += "contrast curve. Please decrease the maximum annular distance"
            raise ValueError(msg)

        # find closest contrast values to requested radii
        ini_contrast = []
        for aa, ad in enumerate(an_dist):
            idx = find_nearest(ini_rads, ad)
            ini_contrast.append(ini_cc[idx])

    if verbose:
        print("Calculating initial SNR map with no injected companion...")

    # TODO: Clean below?
    # Consider 3 cases depending on whether algo is (i) defined externally,
    # (ii) a VIP postproc algorithm; (iii) ineligible for contrast curves
    argl = getfullargspec(algo).args
    if "cube" in argl and "angle_list" in argl and "verbose" in argl:
        # (i) external algorithm with appropriate parameters [OK]
        pass
    else:
        algo_name = algo.__name__
        idx = algo.__module__.index(
            '.', algo.__module__.index('.') + 1)
        mod = algo.__module__[:idx]
        tmp = __import__(
            mod, fromlist=[algo_name.upper()+'_Params'])
        algo_params = getattr(tmp, algo_name.upper()+'_Params')
        argl = [attr for attr in vars(algo_params)]
        if "cube" in argl and "angle_list" in argl and "verbose" in argl:
            # (ii) a VIP postproc algorithm [OK]
            pass
        else:
            # (iii) ineligible routine for contrast curves [Raise error]
            msg = "Ineligible algo for contrast curve function. algo should "
            msg += "have parameters 'cube', 'angle_list' and 'verbose'"
            raise TypeError(msg)

    if "cube" in argl and "angle_list" in argl:
        if "fwhm" in argl:
            frame_fin = algo(
                cube=cube,
                angle_list=angle_list,
                fwhm=fwhm_med,
                verbose=False,
                **algo_dict,
            )
        else:
            frame_fin = algo(
                cube=cube, angle_list=angle_list, verbose=False, **algo_dict
            )
    else:
        raise ValueError("'cube' and 'angle_list' must be arguments of algo")

    snrmap_empty = snrmap(
        frame_fin,
        fwhm,
        approximated=snr_approximation,
        plot=False,
        known_sources=None,
        nproc=nproc,
        array2=None,
        use2alone=False,
        exclude_negative_lobes=False,
        verbose=False,
    )

    cont_curve = np.zeros((len(an_dist)))

    # We crop the PSF and check if PSF has been normalized (so that flux in
    # 1*FWHM aperture = 1) and fix if needed
    new_psf_size = int(round(3 * fwhm_med))
    if new_psf_size % 2 == 0:
        new_psf_size += 1
    # Normalize psf
    psf = normalize_psf(
        psf, fwhm=fwhm, verbose=False, size=min(new_psf_size, psf.shape[1])
    )

    for k in range(len(an_dist)):
        a = an_dist[k]
        level = ini_contrast[k]
        pos_detect = []

        if verbose:
            print("*** Calculating contrast at r = {} ***".format(a))

        detect_bound = [None, None]
        level_bound = [None, None]
        ii = 0
        err_msg = "Could not converge on a contrast level matching required "
        err_msg += "completeness within {} iterations. Tested level: {}. "
        err_msg += "Is there too much self-subtraction? Consider decreasing "
        err_msg += "ncomp if using PCA, or increasing minimum requested radius."

        while len(pos_detect) == 0 and ii < max_iter:
            pos_detect = []
            pos_non_detect = []
            val_detect = []
            val_non_detect = []

            res = pool_map(
                nproc,
                _estimate_snr_fc,
                a,
                iterable(range(0, n_fc)),
                level,
                n_fc,
                cube,
                psf,
                angle_list,
                fwhm,
                algo,
                algo_dict,
                snrmap_empty,
                starphot,
                approximated=snr_approximation,
            )

            for res_i in res:
                if res_i[0] > 0:
                    pos_detect.append(res_i[1])
                    val_detect.append(res_i[0])
                else:
                    pos_non_detect.append(res_i[1])
                    val_non_detect.append(res_i[0])

            if len(pos_detect) == 0:
                level = level * 1.5
            ii += 1

        if verbose:
            msg = "Found contrast level for first TP detection: {}"
            print(msg.format(level))

        if ii == max_iter:
            raise ValueError(err_msg.format(max_iter, level))

        if len(pos_detect) > round(completeness * n_fc):
            detect_bound[1] = len(pos_detect)
            level_bound[1] = level
        elif len(pos_detect) < round(completeness * n_fc):
            detect_bound[0] = len(pos_detect)
            level_bound[0] = level
            pos_non_detect_temp = pos_non_detect.copy()
            val_non_detect_temp = val_non_detect.copy()
            pos_detect_temp = pos_detect.copy()
            val_detect_temp = val_detect.copy()

        cond1 = detect_bound[0] is None or detect_bound[1] is None
        cond2 = len(pos_detect) != round(completeness * n_fc)

        ii = 0
        while cond1 and cond2 and ii < max_iter:
            if detect_bound[0] is None:
                level = level * 0.5
                pos_detect = []
                pos_non_detect = []
                val_detect = []
                val_non_detect = []

                res = pool_map(
                    nproc,
                    _estimate_snr_fc,
                    a,
                    iterable(range(0, n_fc)),
                    level,
                    n_fc,
                    cube,
                    psf,
                    angle_list,
                    fwhm,
                    algo,
                    algo_dict,
                    snrmap_empty,
                    starphot,
                    approximated=snr_approximation,
                )

                for res_i in res:
                    if res_i[0] > 0:
                        pos_detect.append(res_i[1])
                        val_detect.append(res_i[0])
                    else:
                        pos_non_detect.append(res_i[1])
                        val_non_detect.append(res_i[0])

                comp_temp = round(completeness * n_fc)
                if len(pos_detect) > comp_temp and level_bound[1] > level:
                    detect_bound[1] = len(pos_detect)
                    level_bound[1] = level
                elif len(pos_detect) < comp_temp:
                    detect_bound[0] = len(pos_detect)
                    level_bound[0] = level
                    pos_non_detect_temp = pos_non_detect.copy()
                    val_non_detect_temp = val_non_detect.copy()
                    pos_detect_temp = pos_detect.copy()
                    val_detect_temp = val_detect.copy()

            elif detect_bound[1] is None:
                level = level * 1.5
                res = pool_map(
                    nproc,
                    _estimate_snr_fc,
                    a,
                    iterable(-np.sort(-np.array(pos_non_detect))),
                    level,
                    n_fc,
                    cube,
                    psf,
                    angle_list,
                    fwhm,
                    algo,
                    algo_dict,
                    snrmap_empty,
                    starphot,
                    approximated=snr_approximation,
                )

                it = len(pos_non_detect) - 1
                for res_i in res:
                    if res_i[0] > 0:
                        pos_detect.append(res_i[1])
                        val_detect.append(res_i[0])
                        del pos_non_detect[it]
                        del val_non_detect[it]
                    it -= 1

                comp_temp = round(completeness * n_fc)
                if len(pos_detect) > comp_temp:
                    detect_bound[1] = len(pos_detect)
                    level_bound[1] = level
                elif len(pos_detect) < comp_temp and level_bound[0] < level:
                    detect_bound[0] = len(pos_detect)
                    level_bound[0] = level
                    pos_non_detect_temp = pos_non_detect.copy()
                    val_non_detect_temp = val_non_detect.copy()
                    pos_detect_temp = pos_detect.copy()
                    val_detect_temp = val_detect.copy()

            cond1 = detect_bound[0] is None or detect_bound[1] is None
            cond2 = len(pos_detect) != comp_temp
            ii += 1

        if verbose:
            msg = "Found lower and upper bounds of sought contrast: {}"
            print(msg.format(level_bound))

        if ii == max_iter:
            raise ValueError(err_msg.format(max_iter, level))

        if len(pos_detect) != round(completeness * n_fc):
            pos_non_detect = pos_non_detect_temp.copy()
            val_non_detect = val_non_detect_temp.copy()
            pos_detect = pos_detect_temp.copy()
            val_detect = val_detect_temp.copy()

        ii = 0
        while len(pos_detect) != round(completeness * n_fc) and ii < max_iter:
            fact = (level_bound[1] - level_bound[0]) / (
                detect_bound[1] - detect_bound[0]
            )
            level = level_bound[0] + fact * \
                (completeness * n_fc - detect_bound[0])

            res = pool_map(
                nproc,
                _estimate_snr_fc,
                a,
                iterable(-np.sort(-np.array(pos_non_detect))),
                level,
                n_fc,
                cube,
                psf,
                angle_list,
                fwhm,
                algo,
                algo_dict,
                snrmap_empty,
                starphot,
                approximated=snr_approximation,
            )

            it = len(pos_non_detect) - 1
            for res_i in res:
                if res_i[0] > 0:
                    pos_detect.append(res_i[1])
                    val_detect.append(res_i[0])
                    del pos_non_detect[it]
                    del val_non_detect[it]
                it -= 1

            comp_temp = round(completeness * n_fc)
            if len(pos_detect) > comp_temp:
                detect_bound[1] = len(pos_detect)
                level_bound[1] = level
            elif len(pos_detect) < comp_temp and level_bound[0] < level:
                detect_bound[0] = len(pos_detect)
                level_bound[0] = level
                pos_non_detect_temp = pos_non_detect.copy()
                val_non_detect_temp = val_non_detect.copy()
                pos_detect_temp = pos_detect.copy()
                val_detect_temp = val_detect.copy()

            if len(pos_detect) != comp_temp:
                pos_non_detect = pos_non_detect_temp.copy()
                val_non_detect = val_non_detect_temp.copy()
                pos_detect = pos_detect_temp.copy()
                val_detect = val_detect_temp.copy()
            ii += 1

        if ii == max_iter:
            raise ValueError(err_msg.format(max_iter, level))

        if verbose:
            msg = "=> found final contrast for {}% completeness: {}"
            print(msg.format(completeness * 100, level))
        cont_curve[k] = level

    # plotting
    if plot:
        an_dist_arcsec = np.asarray(an_dist) * pxscale
        label = ["Sensitivity"]

        plt.rc("savefig", dpi=dpi)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax1 = fig.add_subplot(111)
        (con1,) = ax1.plot(
            an_dist_arcsec, cont_curve, "-", alpha=0.2, lw=2, color="green"
        )
        (con2,) = ax1.plot(an_dist_arcsec, cont_curve, ".", alpha=0.2, color="green")

        lege = [(con1, con2)]

        plt.legend(lege, label, fancybox=True, fontsize="medium")
        plt.xlabel("Angular separation [arcsec]")
        plt.ylabel(str(int(completeness * 100)) + "% completeness contrast")
        plt.grid("on", which="both", alpha=0.2, linestyle="solid")
        ax1.set_yscale("log")
        ax1.set_xlim(0, 1.1 * np.max(an_dist_arcsec))

        # Give a title to the contrast curve plot
        if object_name is not None:
            # Retrieve ncomp and pca_type info to use in title
            ncomp = algo_dict["ncomp"]
            if algo_dict["cube_ref"] is None:
                pca_type = "ADI"
            else:
                pca_type = "RDI"
            title = "{} {} {}pc".format(pca_type, object_name, ncomp)
            plt.title(title, fontsize=14)

        # Option to fix the y-limit
        if len(fix_y_lim) == 2:
            min_y_lim = min(fix_y_lim[0], fix_y_lim[1])
            max_y_lim = max(fix_y_lim[0], fix_y_lim[1])
            ax1.set_ylim(min_y_lim, max_y_lim)

        # Optionally, save the figure to a path
        if save_plot is not None:
            fig.savefig(save_plot, dpi=dpi)

    return an_dist, cont_curve


def completeness_curve_stim(
    cube,
    angle_list,
    psf,
    fwhm,
    algo,
    an_dist=None,
    ini_contrast=None,
    starphot=1,
    transmission=None,
    pxscale=0.1,
    n_fc=20,
    completeness=0.95,
    conv=False,
    sigma=None,
    Zthresh=None,
    snr_approximation=True,
    max_iter=20,
    precision=0.1,
    through_thresh=0,
    progressive_thr=True,
    width = 1.5,
    mask=None,
    check_max=False,
    algo_dict={},
    verbose=True,
    plot=True,
    dpi=vip_figdpi,
    save_plot=None,
    object_name=None,
    fix_y_lim=(),
    imlib='vip-fft',
    nproc=None,
    figsize=vip_figsize,
    algo_class=None,
):
    """
    Function allowing the computation of completeness-based contrast curves with
    any of the psf-subtraction algorithms provided by VIP. The code relies on
    the approach proposed in [DAH21b]_, itself inspired by the framework
    developed in [JEN18]_. It relies on the computation of the contrast
    associated to a completeness level achieved at a level defined as the first
    false positive in the original SNR map (brightest speckle observed in the
    empty map) instead of the computation o the local noise and throughput (see
    the ``vip_hci.metrics.contrast_curve`` function). The computation of the
    completeness level associated to a contrast is done via the sequential
    injection of multiple fake companions. The algorithm uses multiple
    interpolations to find the contrast associated to the selected completeness
    level (0.95 by default). More information about the algorithm can be found
    in [DAH21b]_.

    Parameters
    ----------
    cube : 3d or 4d numpy ndarray
        The input cube, 3d (ADI data) or 4d array (IFS data), without fake
        companions.
    angle_list : 1d numpy ndarray
        Vector of derotation angles to align North up in your cube images.
    psf : 2d or 3d numpy ndarray
        Frame with the psf template for the fake companion(s).
        PSF must be centered in array. Normalization is done internally.
    fwhm: int or float or 1d array, optional
        The the Full Width Half Maximum in pixels. It can handle a different
        FWHM value for different wavelengths (IFS data).
    algo : callable or function
        The post-processing algorithm, e.g. ``vip_hci.pca.pca``.
    an_dist: list or ndarray, optional
        List of angular separations for which a contrast has to be estimated.
        Default is None which corresponds to a range spanning 2 FWHM to half
        the size of the provided cube - PSF size //2 with a step of 5 pixels
    ini_contrast: list, 1d ndarray or None, optional
        Initial contrast for the range of angular separations included in
        `an_dist`. The number of initial contrasts should be equivalent to the
        number of angular separations. Default is None which corresponds to the
        5-sigma contrast_curve obtained with ``vip_hci.metrics.contrast_curve``.
    starphot : int or float or 1d array, optional
        If int or float it corresponds to the aperture photometry of the
        non-coronagraphic PSF which we use to scale the contrast.
        Default is 1, which corresponds to an output contrast expressed in ADU.
    pxscale : float, optional
        Plate scale or pixel scale of the instrument. Only used for plots.
    n_fc: int, optional
        Number of azimuths considered for the computation of the True
        positive rate/completeness,(number of fake companions injected
        sequentially). The number of azimuths is defined such that the
        selected completeness is reachable (e.g. 95% of completeness
        requires at least 20 fake companion injections). Default 20.
    completeness: float, optional
        The completeness level to be achieved when computing the contrasts,
        i.e. the True positive rate reached at the threshold associated to
        the first false positive (the first false positive is defined as
        the brightest speckle present in the entire detection map).
        Default 95.
    snr_approximation : bool, optional
        If True, an approximated S/N map is generated. If False the
        approach of [MAW14]_ is used. Default is True, for speed.
    max_iter: int, optional
        Maximum number of iterations to consider in the search for the contrast
        level achieving desired completeness before considering it unachievable.
    nproc : int or None, optional
        Number of processes for parallel computing.
    algo_dict: dictionary, optional
        Any other valid parameter of the post-processing algorithms can be
        passed here, including e.g. imlib and interpolation.
    verbose : bool, optional
        Whether to print more info while running the algorithm. Default: True.
    plot : bool, optional
        Whether to plot the final contrast curve or not. True by default.
    dpi : int optional
        Dots per inch for the plots. 100 by default. 300 for printing quality.
    save_plot: string or None, optional
        If provided, the contrast curve will be saved to this path.
    object_name: string or None, optional
        Target name, used in the plot title.
    fix_y_lim: tuple, optional
        If provided, the y axis limits will be fixed, for easier comparison
        between plots.
    fig_size: tuple, optional
        Figure size

    Returns
    -------
    an_dist: 1d numpy ndarray
        Radial distances where the contrasts are calculated
    cont_curve: 1d numpy ndarray
        Contrasts for the considered radial distances and selected completeness
        level.
    """
    
    if (100 * completeness) % (100 / n_fc) > 0:
        n_fc = int(100 / gcd(int(100 * completeness), 100))

    if isinstance(cube, list):
        if cube[0].ndim != 3:
            raise TypeError("The input array is not a list of 3d cubes")
    else:
        if cube.ndim != 3 and cube.ndim != 4:
            raise TypeError("The input array is not a 3d or 4d cube")
        if cube.ndim == 3 and (cube.shape[0] != angle_list.shape[0]):
            raise TypeError("Input parallactic angles vector has wrong length")
        if cube.ndim == 4 and (cube.shape[1] != angle_list.shape[0]):
            raise TypeError("Input parallactic angles vector has wrong length")
        if cube.ndim == 3 and psf.ndim != 2:
            raise TypeError("Template PSF is not a frame (for ADI case)")
        if cube.ndim == 4 and psf.ndim != 3:
            raise TypeError("Template PSF is not a cube (for ADI+IFS case)")
    if nproc is None:
        nproc = cpu_count() // 2

    if isinstance(fwhm, (np.ndarray, list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if an_dist is None:
        raise TypeError("Please define the distances")

    if ini_contrast is None:
        print("Contrast curve not provided => will be computed first...")
        ini_cc = contrast_curve(
            cube,
            angle_list,
            psf,
            fwhm_med,
            pxscale,
            starphot,
            algo,
            transmission=transmission,
            sigma=3,
            nbranch=1,
            theta=0,
            inner_rad=1,
            wedge=(0, 360),
            fc_snr=100,
            plot=False,
            algo_class=algo_class,
            **algo_dict,
        )
        ini_rads = np.array(ini_cc["distance"])
        ini_cc = np.array(ini_cc["sensitivity_student"])

        if np.amax(an_dist) > np.amax(ini_rads):
            msg = "Max requested annular distance larger than covered by "
            msg += "contrast curve. Please decrease the maximum annular distance"
            raise ValueError(msg)

        # find closest contrast values to requested radii
        ini_contrast = []
        for aa, ad in enumerate(an_dist):
            idx = find_nearest(ini_rads, ad)
            ini_contrast.append(ini_cc[idx])


    # TODO: Clean below?
    # Consider 3 cases depending on whether algo is (i) defined externally,
    # (ii) a VIP postproc algorithm; (iii) ineligible for contrast curves
    argl = getfullargspec(algo).args
    if "cube" in argl and "angle_list" in argl and "verbose" in argl:
        # (i) external algorithm with appropriate parameters [OK]
        pass
    else:
        algo_name = algo.__name__
        idx = algo.__module__.index(
            '.', algo.__module__.index('.') + 1)
        mod = algo.__module__[:idx]
        tmp = __import__(
            mod, fromlist=[algo_name.upper()+'_Params'])
        algo_params = getattr(tmp, algo_name.upper()+'_Params')
        argl = [attr for attr in vars(algo_params)]
        if "cube" in argl and "angle_list" in argl and "verbose" in argl:
            # (ii) a VIP postproc algorithm [OK]
            pass
        else:
            # (iii) ineligible routine for contrast curves [Raise error]
            msg = "Ineligible algo for contrast curve function. algo should "
            msg += "have parameters 'cube', 'angle_list' and 'verbose'"
            raise TypeError(msg)

    if "cube" in argl and "angle_list" in argl:
        if algo.__name__ == 'pca':
            output = algo(cube=cube,
                          angle_list=angle_list,
                          full_output = True,
                          **algo_dict)
            
            if len(cube.shape) == 4:
                if 'adimsdi' not in algo_dict.keys():
                    algo_dict['adimsdi'] = Adimsdi.SINGLE
                if 'cube_ref' not in algo_dict.keys():
                    algo_dict['cube_ref'] = None
                if 'scale_list' not in algo_dict.keys():
                    algo_dict['scale_list'] = None
                    
                if algo_dict['scale_list'] is None:
                    frames = output[0]
                    residuals = output[3]
                else:
                    if (algo_dict['adimsdi'] == Adimsdi.DOUBLE or 
                                       algo_dict['cube_ref'] is not None):
                        frames = output[0]
                        residuals = output[1]
                    else:
                        frames = output[0]
                        residuals = output[2]
                
                
                to_collapse = False
                if algo_dict['cube_ref'] is not None:
                    to_collapse = True
                if algo_dict['scale_list'] is None:
                    to_collapse = True
                    
                if to_collapse:
                    residuals = get_adi_res(residuals)
            else:
                frames = output[0]
                residuals = output[3]
                
            size = cube.shape[-1]
        elif algo.__name__ == 'pca_annular':
            output = algo(cube=cube,
                          angle_list=angle_list,
                          verbose=False,
                          full_output = True,
                          **algo_dict)
            
            residuals = output[0]
            frames = output[2]
            size = cube.shape[-1]
        elif '4S' in algo.__name__ or 'FourS' in algo.__name__:
            if isinstance(cube, list) or (isinstance(cube, np.ndarray) and len(cube.shape) == 4):
                if isinstance(cube, list):
                    nch = len(cube)
                    size = cube[0].shape[-1]
                else:
                    nch = cube.shape[0]
                    size = cube.shape[-1]
                if isinstance(angle_list, list):
                    opp_angles = [-L for L in angle_list]
                else:
                    opp_angles = [-angle_list for i in range(nch)]
            else:
                opp_angles = -angle_list
                size = cube.shape[-1]
            output = algo(cube=cube, angle_list=opp_angles, 
                             **algo_dict)
            
            frames = output[2]
            residuals_ = output[1]
        else:
            raise ValueError("algorithm not supported")
    else:
        raise ValueError("'cube' and 'angle_list' must be arguments of algo")

    if 'pca' in algo.__name__:
        ncomp = algo_dict['ncomp']
        if np.isscalar(ncomp):
            ncomp = np.array([ncomp])
        else:
            ncomp = np.array(ncomp)
    
        nncomp = len(ncomp)
        
        stim_threshold = []
        
        if nncomp == 1:
            residuals = residuals.reshape(1,residuals.shape[0], 
                                        residuals.shape[1],residuals.shape[2])
            frames = frames.reshape(1, frames.shape[0], frames.shape[1])
        
        for i,n in enumerate(ncomp):
            this_inverse = inverse_stim_map(residuals[i], angle_list, 
                            imlib=imlib, nproc = nproc)
            
            this_inverseC = this_inverse.copy()
            
            if conv:
                this_inverse = masked_gaussian_convolution(this_inverse, mask, fwhm)
            
            if mask is not None:
                if np.isscalar(mask):
                    this_inverse = mask_circle(this_inverse, mask)
                else:
                    this_inverse *= mask
                    
                pxl_mask = np.where((mask == 1) & (this_inverse is not None))
            else:
                pxl_mask = np.where(this_inverse is not None)
                
                
            if progressive_thr:
                this_max, this_mean, this_std = make_stim2D_threshold(this_inverse, fwhm, width, mask)
            else:
                this_max = np.nanmax(this_inverse)
                this_mean = np.nanmean(this_inverse)
                this_std = np.nanstd(this_inverse)
                
                
            y, x = frames[i].shape
            twopi = 2 * np.pi
            yy = np.zeros((len(an_dist), n_fc))
            xx = np.zeros((len(an_dist), n_fc))
            fluxes = np.zeros((len(an_dist), n_fc))
            for k,a in enumerate(an_dist):
                for b in range(n_fc):
                    sigposy = y / 2 + np.sin(b / n_fc * twopi) * a
                    sigposx = x / 2 + np.cos(b / n_fc * twopi) * a
                    
                    yy[k,b] = sigposy
                    xx[k,b] = sigposx
                    
                apertures = CircularAperture(np.array((xx[k], yy[k])).T, np.mean(fwhm) / 2)
                these_fluxes = aperture_photometry(frames[i], apertures)
                these_fluxes = np.array(these_fluxes["aperture_sum"])
                fluxes[k] = these_fluxes
            
            stim_threshold.append([this_max, np.mean(this_inverse[pxl_mask]), 
                                   np.std(this_inverse[pxl_mask]), 1, fluxes])
            
            if sigma is not None:
                stim_threshold[i][3] = (this_mean + sigma * this_std)/this_max
                
            
    elif '4S' in algo.__name__ or 'FourS' in algo.__name__:
        this_inverse = stim_map(residuals_)
        
        this_inverseC = this_inverse.copy()
        
        if conv:
            this_inverse = masked_gaussian_convolution(this_inverse, mask, fwhm)
        
        if mask is not None:
            if np.isscalar(mask):
                this_inverse = mask_circle(this_inverse, mask)
            else:
                this_inverse *= mask
                
            pxl_mask = np.where((mask == 1) & (this_inverse is not None))
        else:
            pxl_mask = np.where(this_inverse is not None)
            
        if progressive_thr:
            this_max, this_mean, this_std = make_stim2D_threshold(this_inverse, fwhm, width, mask)
        else:
            this_max = np.nanmax(this_inverse)
            this_mean = np.nanmean(this_inverse)
            this_std = np.nanstd(this_inverse)
            
            
        y, x = frames.shape
        twopi = 2 * np.pi
        yy = np.zeros((len(an_dist), n_fc))
        xx = np.zeros((len(an_dist), n_fc))
        fluxes = np.zeros((len(an_dist), n_fc))
        for k,a in enumerate(an_dist):
            for b in range(n_fc):
                sigposy = y / 2 + np.sin(b / n_fc * twopi) * a
                sigposx = x / 2 + np.cos(b / n_fc * twopi) * a
                
                yy[k,b] = sigposy
                xx[k,b] = sigposx
                
            apertures = CircularAperture(np.array((xx[k], yy[k])).T, np.mean(fwhm) / 2)
            these_fluxes = aperture_photometry(frames, apertures)
            these_fluxes = np.array(these_fluxes["aperture_sum"])
            fluxes[k] = these_fluxes
        
        stim_threshold = []
        stim_threshold.append([this_max, np.mean(this_inverse[pxl_mask]), 
                               np.std(this_inverse[pxl_mask]), 1, fluxes])
        
        if sigma is not None:
            stim_threshold[0][3] = (this_mean + sigma * this_std)/this_max


    completeness_curve = np.ones((len(an_dist), 3))

    # We crop the PSF and check if PSF has been normalized (so that flux in
    # 1*FWHM aperture = 1) and fix if needed
    new_psf_size = int(round(3 * fwhm_med))
    if new_psf_size % 2 == 0:
        new_psf_size += 1
    # Normalize psf
    if isinstance(cube, np.ndarray) and len(cube.shape) == 3:
        psf = normalize_psf(
            psf, fwhm=fwhm, verbose=False, size=min(new_psf_size, psf.shape[1])
            )
    else:
        nch = len(cube)
        V = [normalize_psf(psf[i], fwhm[i], size=20, imlib='ndimage-fourier', force_odd = True, full_output = True) for i in range(0, nch, 1)]
        psf, _, _ = [], [], []
        for i in range(0, nch, 1):
            psf.append(V[i][0])
        psf = np.array(psf)
    
    nbr_to_detect = int(round(completeness * n_fc))
    max_missed = int(n_fc - nbr_to_detect)
    
    
    #if transmission is None:
    #    transmission_factors = np.ones(len(an_dist))
    #else:
    #    transmission_factors = np.interp(an_dist, transmission[0], transmission[1], right = 1, left = 0)
    

    for k in range(len(an_dist)):
        a = an_dist[k]
        level = ini_contrast[k]
        
        stop_thr = level/precision
        prev = 0
        pos_detect = []

        if verbose:
            print("*** Calculating contrast at r = {} ***".format(a))

        level_bound = [None, None]
        cond = level_bound[0] is None or level_bound[1] is None
        ii = 0
        err_msg = "Could not converge on a contrast level matching required "
        err_msg += "completeness within {} iterations. Tested level: {}. "
        err_msg += "Is there too much self-subtraction? Consider decreasing "
        err_msg += "ncomp if using PCA, or increasing minimum requested radius."

        while ii < max_iter:
            pos_detect = []
            pos_non_detect = []
            val_detect = []
            val_non_detect = []
            stim_maps = np.zeros((n_fc, size, size))
            
            cond = level_bound[0] is None or level_bound[1] is None
            if verbose and not cond:
                print('Current contrast bounds: ', level_bound)
            
            res = np.zeros((n_fc,2))
            for b in range(0,n_fc):
                this_result = _stim_fc(a,an_dist,b,level, n_fc, cube, psf, angle_list, 
                        fwhm, algo, algo_dict, stim_threshold, through_thresh, 
                        mask, conv, starphot, transmission, False, check_max, this_inverseC, 
                        sigma,Zthresh,width)
                
                res[b] = this_result[0:2]
                stim_maps[b] = this_result[2]
                
                if res[b][0] <= 0:
                    pos_non_detect.append(res[b][1])
                    val_non_detect.append(res[b][0])
                    
                    if len(pos_non_detect) > max_missed:
                        level_bound[0] = level
                        if level_bound[0] == 1:
                            print('No contrast lower than 1 found')
                            ii = max_iter
                            break
                        if level_bound[1] is None:
                            prev = level
                            level *= 1.5
                            if level > 1:
                                level = 1
                            stop_thr = level*precision
                        else:
                            prev = level
                            level = np.mean(level_bound)
                            stop_thr = level*precision
                        break
                else:
                    pos_detect.append(res[b][1])
                    val_detect.append(res[b][0])
                    
                    if len(pos_detect) >= nbr_to_detect:
                        level_bound[1] = level
                        if level_bound[0] is None:
                            prev = level
                            level *= 0.75
                            stop_thr = level*precision
                        else:
                            prev = level
                            level = np.mean(level_bound)
                            stop_thr = level*precision
                        break
            
            cond = level_bound[0] is None or level_bound[1] is None
            if not cond:
                #plot_frames(stim_maps, rows = 5)
                #if np.abs(level_bound[1] - level_bound[0]) < precision*2*10**(np.floor(np.log10(np.abs(level)))):
                if np.abs(level_bound[1] - level_bound[0]) < stop_thr*2:
                    if verbose:
                        print('Precision reached: ', level_bound)
                    break
                ii += 1


        completeness_curve[k,0] = level
        
        if level_bound[0] is None:
            level_bound[0] = 1
        if level_bound[1] is None:
            level_bound[1] = 1
        
        completeness_curve[k,1:] = level_bound

        if ii >= max_iter:
            if level_bound[0] == 1:
                print('Check that there is not too much self-subtraction')
            else:
                print('Could not find contrast for the requested precision within \
                  the amount of iterations allocated') 


    # plotting
    if plot:
        an_dist_arcsec = np.asarray(an_dist) * pxscale
        label = ["Sensitivity"]

        plt.rc("savefig", dpi=dpi)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax1 = fig.add_subplot(111)
        (con1,) = ax1.plot(
            an_dist_arcsec, completeness_curve[:,0], "-", alpha=0, lw=2, color="green"
        )
        (con2,) = ax1.plot(an_dist_arcsec, completeness_curve[:,0], ".", alpha=0, color="green")
        
        yerr_lower = completeness_curve[:,0] - completeness_curve[:,1]
        yerr_upper = completeness_curve[:,2] - completeness_curve[:,0]
        ax1.errorbar(an_dist_arcsec, completeness_curve[:,0], alpha=0,
            yerr=[yerr_lower, yerr_upper], fmt='o', color="green", label="Error bars")
        
        ax1.fill_between(an_dist_arcsec, completeness_curve[:,1], completeness_curve[:,2], 
                    color="green", alpha=0.2, label="Error region")

        lege = [(con1, con2)]

        ax1.set_yscale("log")
        ax1.set_xlim(0, 1.1 * np.max(an_dist_arcsec))

        # Give a title to the contrast curve plot
        if object_name is not None:
            # Retrieve ncomp and pca_type info to use in title
            ncomp = algo_dict["ncomp"]
            if algo_dict["cube_ref"] is None:
                pca_type = "ADI"
            else:
                pca_type = "RDI"
            title = "{} {} {}pc".format(pca_type, object_name, ncomp)
            plt.title(title, fontsize=14)

        # Option to fix the y-limit
        if len(fix_y_lim) == 2:
            min_y_lim = min(fix_y_lim[0], fix_y_lim[1])
            max_y_lim = max(fix_y_lim[0], fix_y_lim[1])
            ax1.set_ylim(min_y_lim, max_y_lim)

        # Optionally, save the figure to a path
        if save_plot is not None:
            fig.savefig(save_plot, dpi=dpi)
            
        plt.show()

    return an_dist, completeness_curve


def completeness_curve_snr(
    cube,
    angle_list,
    psf,
    fwhm,
    algo,
    an_dist=None,
    ini_contrast=None,
    starphot=1,
    transmission=None,
    pxscale=0.1,
    n_fc=20,
    completeness=0.95,
    sigma=5,
    snr_approximation=False,
    exclude_negative_lobes=True,
    max_iter=20,
    precision=0.1,
    through_thresh=0,
    progressive_thr=True,
    width = 1.5,
    mask=None,
    algo_dict={},
    verbose=True,
    plot=True,
    dpi=vip_figdpi,
    save_plot=None,
    object_name=None,
    fix_y_lim=(),
    imlib='vip-fft',
    nproc=None,
    figsize=vip_figsize,
    algo_class=None,
):
    """
    Function allowing the computation of completeness-based contrast curves with
    any of the psf-subtraction algorithms provided by VIP. The code relies on
    the approach proposed in [DAH21b]_, itself inspired by the framework
    developed in [JEN18]_. It relies on the computation of the contrast
    associated to a completeness level achieved at a level defined as the first
    false positive in the original SNR map (brightest speckle observed in the
    empty map) instead of the computation o the local noise and throughput (see
    the ``vip_hci.metrics.contrast_curve`` function). The computation of the
    completeness level associated to a contrast is done via the sequential
    injection of multiple fake companions. The algorithm uses multiple
    interpolations to find the contrast associated to the selected completeness
    level (0.95 by default). More information about the algorithm can be found
    in [DAH21b]_.

    Parameters
    ----------
    cube : 3d or 4d numpy ndarray
        The input cube, 3d (ADI data) or 4d array (IFS data), without fake
        companions.
    angle_list : 1d numpy ndarray
        Vector of derotation angles to align North up in your cube images.
    psf : 2d or 3d numpy ndarray
        Frame with the psf template for the fake companion(s).
        PSF must be centered in array. Normalization is done internally.
    fwhm: int or float or 1d array, optional
        The the Full Width Half Maximum in pixels. It can handle a different
        FWHM value for different wavelengths (IFS data).
    algo : callable or function
        The post-processing algorithm, e.g. ``vip_hci.pca.pca``.
    an_dist: list or ndarray, optional
        List of angular separations for which a contrast has to be estimated.
        Default is None which corresponds to a range spanning 2 FWHM to half
        the size of the provided cube - PSF size //2 with a step of 5 pixels
    ini_contrast: list, 1d ndarray or None, optional
        Initial contrast for the range of angular separations included in
        `an_dist`. The number of initial contrasts should be equivalent to the
        number of angular separations. Default is None which corresponds to the
        5-sigma contrast_curve obtained with ``vip_hci.metrics.contrast_curve``.
    starphot : int or float or 1d array, optional
        If int or float it corresponds to the aperture photometry of the
        non-coronagraphic PSF which we use to scale the contrast.
        Default is 1, which corresponds to an output contrast expressed in ADU.
    pxscale : float, optional
        Plate scale or pixel scale of the instrument. Only used for plots.
    n_fc: int, optional
        Number of azimuths considered for the computation of the True
        positive rate/completeness,(number of fake companions injected
        sequentially). The number of azimuths is defined such that the
        selected completeness is reachable (e.g. 95% of completeness
        requires at least 20 fake companion injections). Default 20.
    completeness: float, optional
        The completeness level to be achieved when computing the contrasts,
        i.e. the True positive rate reached at the threshold associated to
        the first false positive (the first false positive is defined as
        the brightest speckle present in the entire detection map).
        Default 95.
    snr_approximation : bool, optional
        If True, an approximated S/N map is generated. If False the
        approach of [MAW14]_ is used. Default is True, for speed.
    max_iter: int, optional
        Maximum number of iterations to consider in the search for the contrast
        level achieving desired completeness before considering it unachievable.
    nproc : int or None, optional
        Number of processes for parallel computing.
    algo_dict: dictionary, optional
        Any other valid parameter of the post-processing algorithms can be
        passed here, including e.g. imlib and interpolation.
    verbose : bool, optional
        Whether to print more info while running the algorithm. Default: True.
    plot : bool, optional
        Whether to plot the final contrast curve or not. True by default.
    dpi : int optional
        Dots per inch for the plots. 100 by default. 300 for printing quality.
    save_plot: string or None, optional
        If provided, the contrast curve will be saved to this path.
    object_name: string or None, optional
        Target name, used in the plot title.
    fix_y_lim: tuple, optional
        If provided, the y axis limits will be fixed, for easier comparison
        between plots.
    fig_size: tuple, optional
        Figure size

    Returns
    -------
    an_dist: 1d numpy ndarray
        Radial distances where the contrasts are calculated
    cont_curve: 1d numpy ndarray
        Contrasts for the considered radial distances and selected completeness
        level.
    """
    
    if (100 * completeness) % (100 / n_fc) > 0:
        n_fc = int(100 / gcd(int(100 * completeness), 100))

    if cube.ndim != 3 and cube.ndim != 4:
        raise TypeError("The input array is not a 3d or 4d cube")
    if cube.ndim == 3 and (cube.shape[0] != angle_list.shape[0]):
        raise TypeError("Input parallactic angles vector has wrong length")
    if cube.ndim == 4 and (cube.shape[1] != angle_list.shape[0]):
        raise TypeError("Input parallactic angles vector has wrong length")
    if cube.ndim == 3 and psf.ndim != 2:
        raise TypeError("Template PSF is not a frame (for ADI case)")
    if cube.ndim == 4 and psf.ndim != 3:
        raise TypeError("Template PSF is not a cube (for ADI+IFS case)")
    if nproc is None:
        nproc = cpu_count() // 2

    if isinstance(fwhm, (np.ndarray, list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if an_dist is None:
        raise TypeError("Please decfine the distances")

    if ini_contrast is None:
        print("Contrast curve not provided => will be computed first...")
        ini_cc = contrast_curve(
            cube,
            angle_list,
            psf,
            fwhm_med,
            pxscale,
            starphot,
            algo,
            transmission=transmission,
            sigma=3,
            nbranch=1,
            theta=0,
            inner_rad=1,
            wedge=(0, 360),
            fc_snr=100,
            plot=False,
            algo_class=algo_class,
            **algo_dict,
        )
        ini_rads = np.array(ini_cc["distance"])
        ini_cc = np.array(ini_cc["sensitivity_student"])

        if np.amax(an_dist) > np.amax(ini_rads):
            msg = "Max requested annular distance larger than covered by "
            msg += "contrast curve. Please decrease the maximum annular distance"
            raise ValueError(msg)

        # find closest contrast values to requested radii
        ini_contrast = []
        for aa, ad in enumerate(an_dist):
            idx = find_nearest(ini_rads, ad)
            ini_contrast.append(ini_cc[idx])


    # TODO: Clean below?
    # Consider 3 cases depending on whether algo is (i) defined externally,
    # (ii) a VIP postproc algorithm; (iii) ineligible for contrast curves
    argl = getfullargspec(algo).args
    if "cube" in argl and "angle_list" in argl and "verbose" in argl:
        # (i) external algorithm with appropriate parameters [OK]
        pass
    else:
        algo_name = algo.__name__
        idx = algo.__module__.index(
            '.', algo.__module__.index('.') + 1)
        mod = algo.__module__[:idx]
        tmp = __import__(
            mod, fromlist=[algo_name.upper()+'_Params'])
        algo_params = getattr(tmp, algo_name.upper()+'_Params')
        argl = [attr for attr in vars(algo_params)]
        if "cube" in argl and "angle_list" in argl and "verbose" in argl:
            # (ii) a VIP postproc algorithm [OK]
            pass
        else:
            # (iii) ineligible routine for contrast curves [Raise error]
            msg = "Ineligible algo for contrast curve function. algo should "
            msg += "have parameters 'cube', 'angle_list' and 'verbose'"
            raise TypeError(msg)

    if "cube" in argl and "angle_list" in argl:
        if algo.__name__ == 'pca':
            output = algo(cube=cube,
                          angle_list=angle_list,
                          verbose=False,
                          full_output = True,
                          **algo_dict)
            
            if len(cube.shape) == 4:
                if 'adimsdi' not in algo_dict.keys():
                    algo_dict['adimsdi'] = Adimsdi.SINGLE
                if 'cube_ref' not in algo_dict.keys():
                    algo_dict['cube_ref'] = None
                if 'scale_list' not in algo_dict.keys():
                    algo_dict['scale_list'] = None
                    
                if algo_dict['scale_list'] is None:
                    frames = output[0]
                    residuals = output[3]
                else:
                    if (algo_dict['adimsdi'] == Adimsdi.DOUBLE or 
                                       algo_dict['cube_ref'] is not None):
                        frames = output[0]
                        residuals = output[1]
                    else:
                        frames = output[0]
                        residuals = output[2]
                
                
                to_collapse = False
                if algo_dict['cube_ref'] is not None:
                    to_collapse = True
                if algo_dict['scale_list'] is None:
                    to_collapse = True
                    
                if to_collapse:
                    residuals = get_adi_res(residuals)
            else:
                frames = output[0]
                
        elif algo.__name__ == 'pca_annular':
            output = algo(cube=cube,
                          angle_list=angle_list,
                          verbose=False,
                          full_output = True,
                          **algo_dict)
            
            frames = output[2]
        elif '4S' in algo.__name__ or 'FourS' in algo.__name__:
            output = algo(cube=cube, angle_list=-angle_list, 
                             **algo_dict)
            
            frames = output[2]
        else:
            raise ValueError("algorithm not supported")
    else:
        raise ValueError("'cube' and 'angle_list' must be arguments of algo")

    
    completeness_curve = np.ones((len(an_dist), 3))

    # We crop the PSF and check if PSF has been normalized (so that flux in
    # 1*FWHM aperture = 1) and fix if needed
    new_psf_size = int(round(3 * fwhm_med))
    if new_psf_size % 2 == 0:
        new_psf_size += 1
    # Normalize psf
    if len(cube.shape) == 3:
        psf = normalize_psf(
            psf, fwhm=fwhm, verbose=False, size=min(new_psf_size, psf.shape[1])
            )
    else:
        nch = cube.shape[0]
        V = [normalize_psf(psf[i], fwhm[i], size=20, imlib='ndimage-fourier', force_odd = True, full_output = True) for i in range(0, nch, 1)]
        psf, _, _ = [], [], []
        for i in range(0, nch, 1):
            psf.append(V[i][0])
        psf = np.array(psf)
    
    nbr_to_detect = int(round(completeness * n_fc))
    max_missed = int(n_fc - nbr_to_detect)
    
    
    #if transmission is None:
    #    transmission_factors = np.ones(len(an_dist))
    #else:
    #    transmission_factors = np.interp(an_dist, transmission[0], transmission[1], right = 1, left = 0)
    

    for k in range(len(an_dist)):
        a = an_dist[k]
        level = ini_contrast[k]
        
        stop_thr = level/precision
        prev = 0
        pos_detect = []

        if verbose:
            print("*** Calculating contrast at r = {} ***".format(a))

        level_bound = [None, None]
        cond = level_bound[0] is None or level_bound[1] is None
        ii = 0
        err_msg = "Could not converge on a contrast level matching required "
        err_msg += "completeness within {} iterations. Tested level: {}. "
        err_msg += "Is there too much self-subtraction? Consider decreasing "
        err_msg += "ncomp if using PCA, or increasing minimum requested radius."

        while ii < max_iter:
            pos_detect = []
            pos_non_detect = []
            val_detect = []
            val_non_detect = []
            snrmaps = np.zeros((n_fc, cube.shape[-1], cube.shape[-1]))
            
            cond = level_bound[0] is None or level_bound[1] is None
            if verbose and not cond:
                print('Current contrast bounds: ', level_bound)
            
            res = np.zeros((n_fc,2))
            for b in range(0,n_fc):
                this_result = _estimate_snr_fc(a, an_dist, b, level, n_fc, cube, psf, angle_list, 
                        fwhm, algo, algo_dict, through_thresh, sigma,
                        exclude_negative_lobes, starphot, snr_approximation, transmission, False)
                
                res[b] = this_result[0:2]
                snrmaps[b] = this_result[2]
                
                if res[b][0] <= 0:
                    pos_non_detect.append(res[b][1])
                    val_non_detect.append(res[b][0])
                    
                    if len(pos_non_detect) > max_missed:
                        level_bound[0] = level
                        if level_bound[0] == 1:
                            print('No contrast lower than 1 found')
                            ii = max_iter
                            break
                        if level_bound[1] is None:
                            prev = level
                            level *= 1.5
                            if level > 1:
                                level = 1
                            stop_thr = level*precision
                        else:
                            prev = level
                            level = np.mean(level_bound)
                            stop_thr = level*precision
                        break
                else:
                    pos_detect.append(res[b][1])
                    val_detect.append(res[b][0])
                    
                    if len(pos_detect) >= nbr_to_detect:
                        level_bound[1] = level
                        if level_bound[0] is None:
                            prev = level
                            level *= 0.75
                            stop_thr = level*precision
                        else:
                            prev = level
                            level = np.mean(level_bound)
                            stop_thr = level*precision
                        break
            
            cond = level_bound[0] is None or level_bound[1] is None
            if not cond:
                #plot_frames(stim_maps, rows = 5)
                #if np.abs(level_bound[1] - level_bound[0]) < precision*2*10**(np.floor(np.log10(np.abs(level)))):
                if np.abs(level_bound[1] - level_bound[0]) < stop_thr*2:
                    if verbose:
                        print('Precision reached: ', level_bound)
                    break
                ii += 1


        completeness_curve[k,0] = level
        
        if level_bound[0] is None:
            level_bound[0] = 1
        if level_bound[1] is None:
            level_bound[1] = 1
        
        completeness_curve[k,1:] = level_bound

        if ii >= max_iter:
            if level_bound[0] == 1:
                print('Check that there is not too much self-subtraction')
            else:
                print('Could not find contrast for the requested precision within \
                  the amount of iterations allocated') 


    # plotting
    if plot:
        an_dist_arcsec = np.asarray(an_dist) * pxscale
        label = ["Sensitivity"]

        plt.rc("savefig", dpi=dpi)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax1 = fig.add_subplot(111)
        (con1,) = ax1.plot(
            an_dist_arcsec, completeness_curve[:,0], "-", alpha=0, lw=2, color="green"
        )
        (con2,) = ax1.plot(an_dist_arcsec, completeness_curve[:,0], ".", alpha=0, color="green")
        
        yerr_lower = completeness_curve[:,0] - completeness_curve[:,1]
        yerr_upper = completeness_curve[:,2] - completeness_curve[:,0]
        ax1.errorbar(an_dist_arcsec, completeness_curve[:,0], alpha=0,
            yerr=[yerr_lower, yerr_upper], fmt='o', color="green", label="Error bars")
        
        ax1.fill_between(an_dist_arcsec, completeness_curve[:,1], completeness_curve[:,2], 
                    color="green", alpha=0.2, label="Error region")

        lege = [(con1, con2)]

        ax1.set_yscale("log")
        ax1.set_xlim(0, 1.1 * np.max(an_dist_arcsec))

        # Give a title to the contrast curve plot
        if object_name is not None:
            # Retrieve ncomp and pca_type info to use in title
            ncomp = algo_dict["ncomp"]
            if algo_dict["cube_ref"] is None:
                pca_type = "ADI"
            else:
                pca_type = "RDI"
            title = "{} {} {}pc".format(pca_type, object_name, ncomp)
            plt.title(title, fontsize=14)

        # Option to fix the y-limit
        if len(fix_y_lim) == 2:
            min_y_lim = min(fix_y_lim[0], fix_y_lim[1])
            max_y_lim = max(fix_y_lim[0], fix_y_lim[1])
            ax1.set_ylim(min_y_lim, max_y_lim)

        # Optionally, save the figure to a path
        if save_plot is not None:
            fig.savefig(save_plot, dpi=dpi)
            
        plt.show()

    return an_dist, completeness_curve

def plot_fc_results(
    cube,
    angle_list,
    psf,
    fwhm,
    algo,
    an_dist=None,
    starphot=1,
    contrast=None,
    n_fc=20,
    conv=False,
    progressive_thr=True,
    width = 1.5,
    mask=None,
    algo_dict={},
    verbose=True,
    plot=True,
    dpi=vip_figdpi,
    save_plot=None,
    object_name=None,
    fix_y_lim=(),
    imlib='vip-fft',
    nproc=None,
    snr=True,
    transmission=None,
    figsize=vip_figsize,
    algo_class=None,
    ):

    if cube.ndim != 3 and cube.ndim != 4:
        raise TypeError("The input array is not a 3d or 4d cube")
    if cube.ndim == 3 and (cube.shape[0] != angle_list.shape[0]):
        raise TypeError("Input parallactic angles vector has wrong length")
    if cube.ndim == 4 and (cube.shape[1] != angle_list.shape[0]):
        raise TypeError("Input parallactic angles vector has wrong length")
    if cube.ndim == 3 and psf.ndim != 2:
        raise TypeError("Template PSF is not a frame (for ADI case)")
    if cube.ndim == 4 and psf.ndim != 3:
        raise TypeError("Template PSF is not a cube (for ADI+IFS case)")
    if nproc is None:
        nproc = cpu_count() // 2

    if isinstance(fwhm, (np.ndarray, list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if an_dist is None:
        max_dist = int((cube.shape[-1]//2)/fwhm)
        an_dist = np.arange(1, max_dist,1)*fwhm
    if np.isscalar(an_dist):
        an_dist = np.array([an_dist])
    if contrast is None:
        contrast = 5e-3/an_dist
    if np.isscalar(contrast):
        contrast = np.array([contrast])


    # TODO: Clean below?
    # Consider 3 cases depending on whether algo is (i) defined externally,
    # (ii) a VIP postproc algorithm; (iii) ineligible for contrast curves
    argl = getfullargspec(algo).args
    if "cube" in argl and "angle_list" in argl and "verbose" in argl:
        # (i) external algorithm with appropriate parameters [OK]
        pass
    else:
        algo_name = algo.__name__
        idx = algo.__module__.index(
            '.', algo.__module__.index('.') + 1)
        mod = algo.__module__[:idx]
        tmp = __import__(
            mod, fromlist=[algo_name.upper()+'_Params'])
        algo_params = getattr(tmp, algo_name.upper()+'_Params')
        argl = [attr for attr in vars(algo_params)]
        if "cube" in argl and "angle_list" in argl and "verbose" in argl:
            # (ii) a VIP postproc algorithm [OK]
            pass
        else:
            # (iii) ineligible routine for contrast curves [Raise error]
            msg = "Ineligible algo for contrast curve function. algo should "
            msg += "have parameters 'cube', 'angle_list' and 'verbose'"
            raise TypeError(msg)

    if "cube" in argl and "angle_list" in argl:
        if algo.__name__ == 'pca':
            output = algo(cube=cube,
                          angle_list=angle_list,
                          verbose=False,
                          full_output = True,
                          **algo_dict)
            
            if len(cube.shape) == 4:
                if 'adimsdi' not in algo_dict.keys():
                    algo_dict['adimsdi'] = Adimsdi.SINGLE
                if 'cube_ref' not in algo_dict.keys():
                    algo_dict['cube_ref'] = None
                if 'scale_list' not in algo_dict.keys():
                    algo_dict['scale_list'] = None
                    
                if algo_dict['scale_list'] is None:
                    frames = output[0]
                    residuals = output[3]
                    residuals_ = output[4]
                else:
                    if (algo_dict['adimsdi'] == Adimsdi.DOUBLE or 
                                       algo_dict['cube_ref'] is not None):
                        frames = output[0]
                        residuals = output[1]
                        residuals_ = output[2]
                    else:
                        frames = output[0]
                        residuals = output[2]
                        residuals_ = output[3]
                
                to_collapse = False
                if algo_dict['cube_ref'] is not None:
                    to_collapse = True
                if algo_dict['scale_list'] is None:
                    to_collapse = True
                    
                if to_collapse:
                    residuals = get_adi_res(residuals)
            else:
                frames = output[0]
                residuals = output[3]
                residuals_ = output[4]
                
        elif algo.__name__ == 'pca_annular':
            output = algo(cube=cube,
                          angle_list=angle_list,
                          verbose=False,
                          full_output = True,
                          **algo_dict)
            
            residuals = output[0]
            residuals_ = output[1]
            frames = output[2]
        elif '4S' in algo.__name__ or 'FourS' in algo.__name__:
            output = algo(cube=cube, angle_list=-angle_list, 
                             **algo_dict)
            
            direct = algo(cube=cube, angle_list=angle_list, 
                             **algo_dict)
            frames = direct[2]
            residuals = direct[1]
            residuals_ = output[1]
        else:
            raise ValueError("algorithm not supported")
    else:
        raise ValueError("'cube' and 'angle_list' must be arguments of algo")

    if 'pca' in algo.__name__:
        ncomp = algo_dict['ncomp']
        if np.isscalar(ncomp):
            ncomp = np.array([ncomp])
        else:
            ncomp = np.array(ncomp)
    
        nncomp = len(ncomp)
        
        stim_threshold = []
        
        if nncomp == 1:
            residuals = residuals.reshape(1,residuals.shape[0], 
                                        residuals.shape[1],residuals.shape[2])
            residuals_ = residuals_.reshape(1,residuals_.shape[0], 
                                        residuals_.shape[1],residuals_.shape[2])
            frames = frames.reshape(1, frames.shape[0], frames.shape[1])
        
        this_stim = np.zeros_like(frames)
        for i,n in enumerate(ncomp):
            this_inverse = inverse_stim_map(residuals[i], angle_list, 
                            imlib=imlib, nproc = nproc)
            
            if conv:
                this_inverse = masked_gaussian_convolution(this_inverse, mask, fwhm)
            
            if mask is not None:
                if np.isscalar(mask):
                    this_inverse = mask_circle(this_inverse, mask)
                else:
                    this_inverse *= mask
                    
                pxl_mask = np.where((mask == 1) & (this_inverse is not None))
            else:
                pxl_mask = np.where(this_inverse is not None)
                
                
            if progressive_thr:
                this_max, this_mean, this_std = make_stim2D_threshold(this_inverse, fwhm, width, mask)
            else:
                this_max = np.nanmax(this_inverse)
                this_mean = np.nanmean(this_inverse)
                this_std = np.nanstd(this_inverse)
                
                
            y, x = frames[i].shape
            twopi = 2 * np.pi
            yy = np.zeros((len(an_dist), n_fc))
            xx = np.zeros((len(an_dist), n_fc))
            fluxes = np.zeros((len(an_dist), n_fc))
            for k,a in enumerate(an_dist):
                for b in range(n_fc):
                    sigposy = y / 2 + np.sin(b / n_fc * twopi) * a
                    sigposx = x / 2 + np.cos(b / n_fc * twopi) * a
                    
                    yy[k,b] = sigposy
                    xx[k,b] = sigposx
                    
                apertures = CircularAperture(np.array((xx[k], yy[k])).T, np.mean(fwhm) / 2)
                these_fluxes = aperture_photometry(frames[i], apertures)
                these_fluxes = np.array(these_fluxes["aperture_sum"])
                fluxes[k] = these_fluxes
            
            stim_threshold.append([this_max, np.mean(this_inverse[pxl_mask]), 
                                   np.std(this_inverse[pxl_mask]), 1, fluxes])
            
            this_stim[i] = stim_map(residuals_[i])/this_max
    
    elif '4S' in algo.__name__ or 'FourS' in algo.__name__:
        this_inverse = stim_map(residuals_)
        
        if conv:
            this_inverse = masked_gaussian_convolution(this_inverse, mask, fwhm)
        
        if mask is not None:
            if np.isscalar(mask):
                this_inverse = mask_circle(this_inverse, mask)
            else:
                this_inverse *= mask
                
            pxl_mask = np.where((mask == 1) & (this_inverse is not None))
        else:
            pxl_mask = np.where(this_inverse is not None)
            
        if progressive_thr:
            this_max, this_mean, this_std = make_stim2D_threshold(this_inverse, fwhm, width, mask)
        else:
            this_max = np.nanmax(this_inverse)
            this_mean = np.nanmean(this_inverse)
            this_std = np.nanstd(this_inverse)
            
        this_stim = stim_map(residuals)/this_max
            
            
        y, x = frames.shape
        twopi = 2 * np.pi
        yy = np.zeros((len(an_dist), n_fc))
        xx = np.zeros((len(an_dist), n_fc))
        fluxes = np.zeros((len(an_dist), n_fc))
        for k,a in enumerate(an_dist):
            for b in range(n_fc):
                sigposy = y / 2 + np.sin(b / n_fc * twopi) * a
                sigposx = x / 2 + np.cos(b / n_fc * twopi) * a
                
                yy[k,b] = sigposy
                xx[k,b] = sigposx
                
            apertures = CircularAperture(np.array((xx[k], yy[k])).T, np.mean(fwhm) / 2)
            these_fluxes = aperture_photometry(frames, apertures)
            these_fluxes = np.array(these_fluxes["aperture_sum"])
            fluxes[k] = these_fluxes
        
        stim_threshold = []
        stim_threshold.append([this_max, np.mean(this_inverse[pxl_mask]), 
                               np.std(this_inverse[pxl_mask]), 1, fluxes])

    # We crop the PSF and check if PSF has been normalized (so that flux in
    # 1*FWHM aperture = 1) and fix if needed
    new_psf_size = int(round(3 * fwhm_med))
    if new_psf_size % 2 == 0:
        new_psf_size += 1
    # Normalize psf
    if len(cube.shape) == 3:
        psf = normalize_psf(
            psf, fwhm=fwhm, verbose=False, size=min(new_psf_size, psf.shape[1])
            )
    else:
        nch = cube.shape[0]
        V = [normalize_psf(psf[i], fwhm[i], size=20, imlib='ndimage-fourier', force_odd = True, full_output = True) for i in range(0, nch, 1)]
        psf, _, _ = [], [], []
        for i in range(0, nch, 1):
            psf.append(V[i][0])
        psf = np.array(psf)
    

    nbr_dist = len(an_dist)
    images = np.zeros((nbr_dist,n_fc, cube.shape[-1], cube.shape[-1]))
    stim_maps = np.zeros((nbr_dist,n_fc, cube.shape[-1], cube.shape[-1]))
    
    if contrast is None:
        contrast = np.full((nbr_dist), 0.001)
    if np.isscalar(contrast):
        contrast = np.full((nbr_dist), contrast)
    
    if snr:
        stim_values = np.zeros((nbr_dist,n_fc))
        throughput = np.zeros((nbr_dist,n_fc))
        snr_values = np.zeros((nbr_dist,n_fc,2))
        snrmaps = np.zeros((nbr_dist, n_fc, cube.shape[-1], cube.shape[-1]))
    
    for k in range(nbr_dist):
        a = an_dist[k]
        level = contrast[k]

        if verbose:
            print("*** Injecting fake companions at r = {} ***".format(a))

        for b in range(0,n_fc):
            this_result = _stim_fc(a,an_dist,b,level, n_fc, cube, psf, angle_list, 
                        fwhm, algo, algo_dict, stim_threshold, 0, 
                        mask, conv, starphot, transmission, snr)
            
            if snr:
                images[k,b] = this_result[3]
                stim_maps[k,b] = this_result[2]
                stim_values[k,b] = this_result[4][0]
                snr_values[k,b] = this_result[5][:]
                snrmaps[k,b] = this_result[6][0]
                throughput[k,b] = this_result[7]
            else:
                images[k,b] = this_result[3]
                stim_maps[k,b] = this_result[2]
                

    if snr:
        return frames, this_stim, images, stim_maps, snrmaps, stim_values, snr_values, throughput
    else:
        return frames, this_stim, images, stim_maps


def completeness_curve_stim_pca(
    cube,
    angle_list,
    psf,
    fwhm,
    algo,
    an_dist=None,
    ini_contrast=None,
    starphot=1,
    pxscale=0.1,
    n_fc=20,
    completeness=0.95,
    snr_approximation=True,
    max_iter=20,
    precision=10,
    mask=None,
    algo_dict={},
    verbose=True,
    plot=True,
    dpi=vip_figdpi,
    save_plot=None,
    object_name=None,
    fix_y_lim=(),
    imlib='vip-fft',
    nproc=None,
    figsize=vip_figsize,
    algo_class=None,
):
    """
    Function allowing the computation of completeness-based contrast curves with
    any of the psf-subtraction algorithms provided by VIP. The code relies on
    the approach proposed in [DAH21b]_, itself inspired by the framework
    developed in [JEN18]_. It relies on the computation of the contrast
    associated to a completeness level achieved at a level defined as the first
    false positive in the original SNR map (brightest speckle observed in the
    empty map) instead of the computation o the local noise and throughput (see
    the ``vip_hci.metrics.contrast_curve`` function). The computation of the
    completeness level associated to a contrast is done via the sequential
    injection of multiple fake companions. The algorithm uses multiple
    interpolations to find the contrast associated to the selected completeness
    level (0.95 by default). More information about the algorithm can be found
    in [DAH21b]_.

    Parameters
    ----------
    cube : 3d or 4d numpy ndarray
        The input cube, 3d (ADI data) or 4d array (IFS data), without fake
        companions.
    angle_list : 1d numpy ndarray
        Vector of derotation angles to align North up in your cube images.
    psf : 2d or 3d numpy ndarray
        Frame with the psf template for the fake companion(s).
        PSF must be centered in array. Normalization is done internally.
    fwhm: int or float or 1d array, optional
        The the Full Width Half Maximum in pixels. It can handle a different
        FWHM value for different wavelengths (IFS data).
    algo : callable or function
        The post-processing algorithm, e.g. ``vip_hci.pca.pca``.
    an_dist: list or ndarray, optional
        List of angular separations for which a contrast has to be estimated.
        Default is None which corresponds to a range spanning 2 FWHM to half
        the size of the provided cube - PSF size //2 with a step of 5 pixels
    ini_contrast: list, 1d ndarray or None, optional
        Initial contrast for the range of angular separations included in
        `an_dist`. The number of initial contrasts should be equivalent to the
        number of angular separations. Default is None which corresponds to the
        5-sigma contrast_curve obtained with ``vip_hci.metrics.contrast_curve``.
    starphot : int or float or 1d array, optional
        If int or float it corresponds to the aperture photometry of the
        non-coronagraphic PSF which we use to scale the contrast.
        Default is 1, which corresponds to an output contrast expressed in ADU.
    pxscale : float, optional
        Plate scale or pixel scale of the instrument. Only used for plots.
    n_fc: int, optional
        Number of azimuths considered for the computation of the True
        positive rate/completeness,(number of fake companions injected
        sequentially). The number of azimuths is defined such that the
        selected completeness is reachable (e.g. 95% of completeness
        requires at least 20 fake companion injections). Default 20.
    completeness: float, optional
        The completeness level to be achieved when computing the contrasts,
        i.e. the True positive rate reached at the threshold associated to
        the first false positive (the first false positive is defined as
        the brightest speckle present in the entire detection map).
        Default 95.
    snr_approximation : bool, optional
        If True, an approximated S/N map is generated. If False the
        approach of [MAW14]_ is used. Default is True, for speed.
    max_iter: int, optional
        Maximum number of iterations to consider in the search for the contrast
        level achieving desired completeness before considering it unachievable.
    nproc : int or None, optional
        Number of processes for parallel computing.
    algo_dict: dictionary, optional
        Any other valid parameter of the post-processing algorithms can be
        passed here, including e.g. imlib and interpolation.
    verbose : bool, optional
        Whether to print more info while running the algorithm. Default: True.
    plot : bool, optional
        Whether to plot the final contrast curve or not. True by default.
    dpi : int optional
        Dots per inch for the plots. 100 by default. 300 for printing quality.
    save_plot: string or None, optional
        If provided, the contrast curve will be saved to this path.
    object_name: string or None, optional
        Target name, used in the plot title.
    fix_y_lim: tuple, optional
        If provided, the y axis limits will be fixed, for easier comparison
        between plots.
    fig_size: tuple, optional
        Figure size

    Returns
    -------
    an_dist: 1d numpy ndarray
        Radial distances where the contrasts are calculated
    cont_curve: 1d numpy ndarray
        Contrasts for the considered radial distances and selected completeness
        level.
    """

    if (100 * completeness) % (100 / n_fc) > 0:
        n_fc = int(100 / gcd(int(100 * completeness), 100))

    if cube.ndim != 3 and cube.ndim != 4:
        raise TypeError("The input array is not a 3d or 4d cube")
    if cube.ndim == 3 and (cube.shape[0] != angle_list.shape[0]):
        raise TypeError("Input parallactic angles vector has wrong length")
    if cube.ndim == 4 and (cube.shape[1] != angle_list.shape[0]):
        raise TypeError("Input parallactic angles vector has wrong length")
    if cube.ndim == 3 and psf.ndim != 2:
        raise TypeError("Template PSF is not a frame (for ADI case)")
    if cube.ndim == 4 and psf.ndim != 3:
        raise TypeError("Template PSF is not a cube (for ADI+IFS case)")
    if nproc is None:
        nproc = cpu_count() // 2

    if isinstance(fwhm, (np.ndarray, list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if an_dist is None:
        an_dist = np.array(
            range(2 * round(fwhm_med),
                  int(cube.shape[-1] // 2 - 2 * fwhm_med), 5)
        )
        print("an_dist not provided, the following list will be used:", an_dist)
    elif an_dist[-1] > cube.shape[-1] // 2 - 2 * fwhm_med:
        raise TypeError("Please decrease the maximum annular distance")

    if ini_contrast is None:
        print("Contrast curve not provided => will be computed first...")
        ini_cc = contrast_curve(
            cube,
            angle_list,
            psf,
            fwhm_med,
            pxscale,
            starphot,
            algo,
            sigma=3,
            nbranch=1,
            theta=0,
            inner_rad=1,
            wedge=(0, 360),
            fc_snr=100,
            plot=False,
            algo_class=algo_class,
            **algo_dict,
        )
        ini_rads = np.array(ini_cc["distance"])
        ini_cc = np.array(ini_cc["sensitivity_student"])

        if np.amax(an_dist) > np.amax(ini_rads):
            msg = "Max requested annular distance larger than covered by "
            msg += "contrast curve. Please decrease the maximum annular distance"
            raise ValueError(msg)

        # find closest contrast values to requested radii
        ini_contrast = []
        for aa, ad in enumerate(an_dist):
            idx = find_nearest(ini_rads, ad)
            ini_contrast.append(ini_cc[idx])


    # TODO: Clean below?
    # Consider 3 cases depending on whether algo is (i) defined externally,
    # (ii) a VIP postproc algorithm; (iii) ineligible for contrast curves
    argl = getfullargspec(algo).args
    if "cube" in argl and "angle_list" in argl and "verbose" in argl:
        # (i) external algorithm with appropriate parameters [OK]
        pass
    else:
        algo_name = algo.__name__
        idx = algo.__module__.index(
            '.', algo.__module__.index('.') + 1)
        mod = algo.__module__[:idx]
        tmp = __import__(
            mod, fromlist=[algo_name.upper()+'_Params'])
        algo_params = getattr(tmp, algo_name.upper()+'_Params')
        argl = [attr for attr in vars(algo_params)]
        if "cube" in argl and "angle_list" in argl and "verbose" in argl:
            # (ii) a VIP postproc algorithm [OK]
            pass
        else:
            # (iii) ineligible routine for contrast curves [Raise error]
            msg = "Ineligible algo for contrast curve function. algo should "
            msg += "have parameters 'cube', 'angle_list' and 'verbose'"
            raise TypeError(msg)

    if "cube" in argl and "angle_list" in argl:
        if algo.__name__ == 'pca':
            output = algo(cube=cube,
                          angle_list=angle_list,
                          verbose=False,
                          full_output = True,
                          **algo_dict)
            
            residuals = output[3]
        elif algo.__name__ == 'pca_annular':
            output = algo(cube=cube,
                          angle_list=angle_list,
                          verbose=False,
                          full_output = True,
                          **algo_dict)
            
            residuals = output[0]
        else:
            raise ValueError("algorithm not supported")
    else:
        raise ValueError("'cube' and 'angle_list' must be arguments of algo")

    if 'pca' in algo.__name__:
        ncomp = algo_dict['ncomp']
        if np.isscalar(ncomp):
            ncomp = np.array([ncomp])
        else:
            ncomp = np.array(ncomp)
    
        nncomp = len(ncomp)
        
        stim_threshold = np.zeros(nncomp)
        
        if nncomp == 1:
            residuals = residuals.reshape(1,residuals.shape[0], 
                                        residuals.shape[1],residuals.shape[2])
        
        for i,n in enumerate(ncomp):
            this_inverse = inverse_stim_map(residuals[i], angle_list, 
                            imlib=imlib, nproc = nproc)
            if mask is not None:
                if np.isscalar(mask):
                    this_inverse = mask_circle(this_inverse, mask)
                else:
                    this_inverse *= mask
            stim_threshold[i] = np.nanmax(this_inverse)

    completeness_curve = np.zeros((len(an_dist), 3))

    # We crop the PSF and check if PSF has been normalized (so that flux in
    # 1*FWHM aperture = 1) and fix if needed
    new_psf_size = int(round(3 * fwhm_med))
    if new_psf_size % 2 == 0:
        new_psf_size += 1
    # Normalize psf
    psf = normalize_psf(
        psf, fwhm=fwhm, verbose=False, size=min(new_psf_size, psf.shape[1])
    )
    
    nbr_to_detect = int(round(completeness * n_fc))
    max_missed = int(n_fc - nbr_to_detect)
    

    saved_dict = algo_dict.copy()
    for k in range(len(an_dist)):
        algo_dict = saved_dict.copy()
        a = an_dist[k]
        level = ini_contrast[k]
        flux_ncomp = [level, ncomp]
        
        stop_thr = level/precision
        prev = 0
        pos_detect = []

        if verbose:
            print("*** Calculating contrast at r = {} ***".format(a))

        level_bound = [None, None]
        cond = level_bound[0] is None or level_bound[1] is None
        ii = 0
        err_msg = "Could not converge on a contrast level matching required "
        err_msg += "completeness within {} iterations. Tested level: {}. "
        err_msg += "Is there too much self-subtraction? Consider decreasing "
        err_msg += "ncomp if using PCA, or increasing minimum requested radius."

        while ii < max_iter:
            pos_detect = []
            pos_non_detect = []
            val_detect = []
            val_non_detect = []
            
            res = np.zeros((n_fc,2))
            for b in range(0,n_fc):
                this_algo_dict = saved_dict.copy()
                this_algo_dict['ncomp'] = flux_ncomp[1]
                this_level = flux_ncomp[0]
                
                res[b] = _stim_fc(a,b,this_level, n_fc, cube, psf, angle_list, 
                        fwhm, algo, this_algo_dict, stim_threshold, starphot)
                
                if res[b][0] <= 0:
                    pos_non_detect.append(res[b][1])
                    val_non_detect.append(res[b][0])
                    
                    if len(pos_non_detect) > max_missed:
                        level_bound[0] = level
                        if level_bound[0] == 1:
                            print('no contrast lower than 1 found')
                            ii = max_iter
                            break
                        if level_bound[1] is None:
                            prev = level
                            level *= 1.5
                            if level > 1:
                                level = 1
                            stop_thr = level/precision
                        else:
                            prev = level
                            level = np.mean(level_bound)
                            stop_thr = level/precision
                        
                        print('gain ', b, ' ', level)
                        break
                else:
                    pos_detect.append(res[b][1])
                    val_detect.append(res[b][0])
                    
                    if len(pos_detect) > nbr_to_detect:
                        level_bound[1] = level
                        if level_bound[0] is None:
                            prev = level
                            level *= 0.75
                            stop_thr = level/precision
                        else:
                            prev = level
                            level = np.mean(level_bound)
                            stop_thr = level/precision
                            
                        print('gain ', b, ' ', level)
                        break
            
            if len(pos_detect) == nbr_to_detect:
                prev = level
                level_bound[1] = level
                cond = level_bound[0] is None or level_bound[1] is None
                if cond:
                    level *= 0.75
                else:
                    level = np.mean(level_bound)
                stop_thr = level/precision
                
            
            cond = level_bound[0] is None or level_bound[1] is None
            if not cond:
                if np.abs(prev - level) < stop_thr:
                    print('precision reached')
                    break
                print('+1')
                ii += 1


        completeness_curve[k,0] = level
        completeness_curve[k,1:] = level_bound

        if ii >= max_iter:
            print('Could not find suitable contrast')


    # plotting
    if plot:
        an_dist_arcsec = np.asarray(an_dist) * pxscale
        label = ["Sensitivity"]

        plt.rc("savefig", dpi=dpi)
        fig = plt.figure(figsize=figsize, dpi=dpi)
        ax1 = fig.add_subplot(111)
        (con1,) = ax1.plot(
            an_dist_arcsec, completeness_curve[:,0], "-", alpha=0, lw=2, color="green"
        )
        (con2,) = ax1.plot(an_dist_arcsec, completeness_curve[:,0], ".", alpha=0, color="green")
        
        yerr_lower = completeness_curve[:,0] - completeness_curve[:,1]
        yerr_upper = completeness_curve[:,2] - completeness_curve[:,0]
        ax1.errorbar(an_dist_arcsec, completeness_curve[:,0], alpha=0,
            yerr=[yerr_lower, yerr_upper], fmt='o', color="green", label="Error bars")
        
        ax1.fill_between(an_dist_arcsec, completeness_curve[:,1], completeness_curve[:,2], 
                    color="green", alpha=0.2, label="Error region")

        lege = [(con1, con2)]

        plt.legend(lege, label, fancybox=True, fontsize="medium")
        plt.xlabel("Angular separation [arcsec]")
        plt.ylabel(str(int(completeness * 100)) + "% completeness contrast")
        plt.grid("on", which="both", alpha=0.2, linestyle="solid")
        ax1.set_yscale("log")
        ax1.set_xlim(0, 1.1 * np.max(an_dist_arcsec))

        # Give a title to the contrast curve plot
        if object_name is not None:
            # Retrieve ncomp and pca_type info to use in title
            ncomp = algo_dict["ncomp"]
            if algo_dict["cube_ref"] is None:
                pca_type = "ADI"
            else:
                pca_type = "RDI"
            title = "{} {} {}pc".format(pca_type, object_name, ncomp)
            plt.title(title, fontsize=14)

        # Option to fix the y-limit
        if len(fix_y_lim) == 2:
            min_y_lim = min(fix_y_lim[0], fix_y_lim[1])
            max_y_lim = max(fix_y_lim[0], fix_y_lim[1])
            ax1.set_ylim(min_y_lim, max_y_lim)

        # Optionally, save the figure to a path
        if save_plot is not None:
            fig.savefig(save_plot, dpi=dpi)
            
        plt.show()

    return an_dist, completeness_curve


def completeness_theta_map(
    cube,
    angle_list,
    psf,
    fwhm,
    algo,
    an_dist=None,
    ini_contrast=None,
    starphot=1,
    transmission=None,
    pxscale=0.1,
    n_fc=20,
    conv=False,
    sigma=None,
    snr_approximation=True,
    max_iter=20,
    precision=0.1,
    through_thresh=0.1,
    progressive_thr=True,
    width = 1.5,
    mask=None,
    algo_dict={},
    verbose=True,
    plot=True,
    dpi=vip_figdpi,
    save_plot=None,
    object_name=None,
    fix_y_lim=(),
    imlib='vip-fft',
    nproc=None,
    figsize=vip_figsize,
    algo_class=None,
):
    """
    Function allowing the computation of completeness-based contrast curves with
    any of the psf-subtraction algorithms provided by VIP. The code relies on
    the approach proposed in [DAH21b]_, itself inspired by the framework
    developed in [JEN18]_. It relies on the computation of the contrast
    associated to a completeness level achieved at a level defined as the first
    false positive in the original SNR map (brightest speckle observed in the
    empty map) instead of the computation o the local noise and throughput (see
    the ``vip_hci.metrics.contrast_curve`` function). The computation of the
    completeness level associated to a contrast is done via the sequential
    injection of multiple fake companions. The algorithm uses multiple
    interpolations to find the contrast associated to the selected completeness
    level (0.95 by default). More information about the algorithm can be found
    in [DAH21b]_.

    Parameters
    ----------
    cube : 3d or 4d numpy ndarray
        The input cube, 3d (ADI data) or 4d array (IFS data), without fake
        companions.
    angle_list : 1d numpy ndarray
        Vector of derotation angles to align North up in your cube images.
    psf : 2d or 3d numpy ndarray
        Frame with the psf template for the fake companion(s).
        PSF must be centered in array. Normalization is done internally.
    fwhm: int or float or 1d array, optional
        The the Full Width Half Maximum in pixels. It can handle a different
        FWHM value for different wavelengths (IFS data).
    algo : callable or function
        The post-processing algorithm, e.g. ``vip_hci.pca.pca``.
    an_dist: list or ndarray, optional
        List of angular separations for which a contrast has to be estimated.
        Default is None which corresponds to a range spanning 2 FWHM to half
        the size of the provided cube - PSF size //2 with a step of 5 pixels
    ini_contrast: list, 1d ndarray or None, optional
        Initial contrast for the range of angular separations included in
        `an_dist`. The number of initial contrasts should be equivalent to the
        number of angular separations. Default is None which corresponds to the
        5-sigma contrast_curve obtained with ``vip_hci.metrics.contrast_curve``.
    starphot : int or float or 1d array, optional
        If int or float it corresponds to the aperture photometry of the
        non-coronagraphic PSF which we use to scale the contrast.
        Default is 1, which corresponds to an output contrast expressed in ADU.
    pxscale : float, optional
        Plate scale or pixel scale of the instrument. Only used for plots.
    n_fc: int, optional
        Number of azimuths considered for the computation of the True
        positive rate/completeness,(number of fake companions injected
        sequentially). The number of azimuths is defined such that the
        selected completeness is reachable (e.g. 95% of completeness
        requires at least 20 fake companion injections). Default 20.
    completeness: float, optional
        The completeness level to be achieved when computing the contrasts,
        i.e. the True positive rate reached at the threshold associated to
        the first false positive (the first false positive is defined as
        the brightest speckle present in the entire detection map).
        Default 95.
    snr_approximation : bool, optional
        If True, an approximated S/N map is generated. If False the
        approach of [MAW14]_ is used. Default is True, for speed.
    max_iter: int, optional
        Maximum number of iterations to consider in the search for the contrast
        level achieving desired completeness before considering it unachievable.
    nproc : int or None, optional
        Number of processes for parallel computing.
    algo_dict: dictionary, optional
        Any other valid parameter of the post-processing algorithms can be
        passed here, including e.g. imlib and interpolation.
    verbose : bool, optional
        Whether to print more info while running the algorithm. Default: True.
    plot : bool, optional
        Whether to plot the final contrast curve or not. True by default.
    dpi : int optional
        Dots per inch for the plots. 100 by default. 300 for printing quality.
    save_plot: string or None, optional
        If provided, the contrast curve will be saved to this path.
    object_name: string or None, optional
        Target name, used in the plot title.
    fix_y_lim: tuple, optional
        If provided, the y axis limits will be fixed, for easier comparison
        between plots.
    fig_size: tuple, optional
        Figure size

    Returns
    -------
    an_dist: 1d numpy ndarray
        Radial distances where the contrasts are calculated
    cont_curve: 1d numpy ndarray
        Contrasts for the considered radial distances and selected completeness
        level.
    """

    if cube.ndim != 3 and cube.ndim != 4:
        raise TypeError("The input array is not a 3d or 4d cube")
    if cube.ndim == 3 and (cube.shape[0] != angle_list.shape[0]):
        raise TypeError("Input parallactic angles vector has wrong length")
    if cube.ndim == 4 and (cube.shape[1] != angle_list.shape[0]):
        raise TypeError("Input parallactic angles vector has wrong length")
    if cube.ndim == 3 and psf.ndim != 2:
        raise TypeError("Template PSF is not a frame (for ADI case)")
    if cube.ndim == 4 and psf.ndim != 3:
        raise TypeError("Template PSF is not a cube (for ADI+IFS case)")
    if nproc is None:
        nproc = cpu_count() // 2

    if isinstance(fwhm, (np.ndarray, list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if an_dist is None:
        raise TypeError("Please decfine the distances")

    if ini_contrast is None:
        print("Contrast curve not provided => will be computed first...")
        ini_cc = contrast_curve(
            cube,
            angle_list,
            psf,
            fwhm_med,
            pxscale,
            starphot,
            algo,
            transmission=transmission,
            sigma=3,
            nbranch=1,
            theta=0,
            inner_rad=1,
            wedge=(0, 360),
            fc_snr=100,
            plot=False,
            algo_class=algo_class,
            **algo_dict,
        )
        ini_rads = np.array(ini_cc["distance"])
        ini_cc = np.array(ini_cc["sensitivity_student"])

        if np.amax(an_dist) > np.amax(ini_rads):
            msg = "Max requested annular distance larger than covered by "
            msg += "contrast curve. Please decrease the maximum annular distance"
            raise ValueError(msg)

        # find closest contrast values to requested radii
        ini_contrast = []
        for aa, ad in enumerate(an_dist):
            idx = find_nearest(ini_rads, ad)
            ini_contrast.append(ini_cc[idx])


    # TODO: Clean below?
    # Consider 3 cases depending on whether algo is (i) defined externally,
    # (ii) a VIP postproc algorithm; (iii) ineligible for contrast curves
    argl = getfullargspec(algo).args
    if "cube" in argl and "angle_list" in argl and "verbose" in argl:
        # (i) external algorithm with appropriate parameters [OK]
        pass
    else:
        algo_name = algo.__name__
        idx = algo.__module__.index(
            '.', algo.__module__.index('.') + 1)
        mod = algo.__module__[:idx]
        tmp = __import__(
            mod, fromlist=[algo_name.upper()+'_Params'])
        algo_params = getattr(tmp, algo_name.upper()+'_Params')
        argl = [attr for attr in vars(algo_params)]
        if "cube" in argl and "angle_list" in argl and "verbose" in argl:
            # (ii) a VIP postproc algorithm [OK]
            pass
        else:
            # (iii) ineligible routine for contrast curves [Raise error]
            msg = "Ineligible algo for contrast curve function. algo should "
            msg += "have parameters 'cube', 'angle_list' and 'verbose'"
            raise TypeError(msg)

    if "cube" in argl and "angle_list" in argl:
        if algo.__name__ == 'pca':
            output = algo(cube=cube,
                          angle_list=angle_list,
                          verbose=False,
                          full_output = True,
                          **algo_dict)
            
            if len(cube.shape) == 4:
                if 'adimsdi' not in algo_dict.keys():
                    algo_dict['adimsdi'] = Adimsdi.SINGLE
                if 'cube_ref' not in algo_dict.keys():
                    algo_dict['cube_ref'] = None
                if 'scale_list' not in algo_dict.keys():
                    algo_dict['scale_list'] = None
                    
                if algo_dict['scale_list'] is None:
                    frames = output[0]
                    residuals = output[3]
                else:
                    if (algo_dict['adimsdi'] == Adimsdi.DOUBLE or 
                                       algo_dict['cube_ref'] is not None):
                        frames = output[0]
                        residuals = output[1]
                    else:
                        frames = output[0]
                        residuals = output[2]
                
                
                to_collapse = False
                if algo_dict['cube_ref'] is not None:
                    to_collapse = True
                if algo_dict['scale_list'] is None:
                    to_collapse = True
                    
                if to_collapse:
                    residuals = get_adi_res(residuals)
            else:
                frames = output[0]
                residuals = output[3]
                
        elif algo.__name__ == 'pca_annular':
            output = algo(cube=cube,
                          angle_list=angle_list,
                          verbose=False,
                          full_output = True,
                          **algo_dict)
            
            residuals = output[0]
            frames = output[2]
        elif '4S' in algo.__name__ or 'FourS' in algo.__name__:
            output = algo(cube=cube, angle_list=-angle_list, 
                             **algo_dict)
            
            frames = output[2]
            residuals_ = output[1]
        else:
            raise ValueError("algorithm not supported")
    else:
        raise ValueError("'cube' and 'angle_list' must be arguments of algo")

    if 'pca' in algo.__name__:
        ncomp = algo_dict['ncomp']
        if np.isscalar(ncomp):
            ncomp = np.array([ncomp])
        else:
            ncomp = np.array(ncomp)
    
        nncomp = len(ncomp)
        
        stim_threshold = []
        
        if nncomp == 1:
            residuals = residuals.reshape(1,residuals.shape[0], 
                                        residuals.shape[1],residuals.shape[2])
            frames = frames.reshape(1, frames.shape[0], frames.shape[1])
        
        for i,n in enumerate(ncomp):
            this_inverse = inverse_stim_map(residuals[i], angle_list, 
                            imlib=imlib, nproc = nproc)
            
            if conv:
                this_inverse = masked_gaussian_convolution(this_inverse, mask, fwhm)
            
            if mask is not None:
                if np.isscalar(mask):
                    this_inverse = mask_circle(this_inverse, mask)
                else:
                    this_inverse *= mask
                    
                pxl_mask = np.where((mask == 1) & (this_inverse is not None))
            else:
                pxl_mask = np.where(this_inverse is not None)
                
                
            if progressive_thr:
                this_max, this_mean, this_std = make_stim2D_threshold(this_inverse, fwhm, width, mask)
            else:
                this_max = np.nanmax(this_inverse)
                this_mean = np.nanmean(this_inverse)
                this_std = np.nanstd(this_inverse)
                
                
            y, x = frames[i].shape
            twopi = 2 * np.pi
            yy = np.zeros((len(an_dist), n_fc))
            xx = np.zeros((len(an_dist), n_fc))
            fluxes = np.zeros((len(an_dist), n_fc))
            for k,a in enumerate(an_dist):
                for b in range(n_fc):
                    sigposy = y / 2 + np.sin(b / n_fc * twopi) * a
                    sigposx = x / 2 + np.cos(b / n_fc * twopi) * a
                    
                    yy[k,b] = sigposy
                    xx[k,b] = sigposx
                    
                apertures = CircularAperture(np.array((xx[k], yy[k])).T, np.mean(fwhm) / 2)
                these_fluxes = aperture_photometry(frames[i], apertures)
                these_fluxes = np.array(these_fluxes["aperture_sum"])
                fluxes[k] = these_fluxes
            
            stim_threshold.append([this_max, np.mean(this_inverse[pxl_mask]), 
                                   np.std(this_inverse[pxl_mask]), 1, fluxes])
            
            if sigma is not None:
                stim_threshold[i][3] = (this_mean + sigma * this_std)/this_max
                
            
    elif '4S' in algo.__name__ or 'FourS' in algo.__name__:
        this_inverse = stim_map(residuals_)
        
        if conv:
            this_inverse = masked_gaussian_convolution(this_inverse, mask, fwhm)
        
        if mask is not None:
            if np.isscalar(mask):
                this_inverse = mask_circle(this_inverse, mask)
            else:
                this_inverse *= mask
                
            pxl_mask = np.where((mask == 1) & (this_inverse is not None))
        else:
            pxl_mask = np.where(this_inverse is not None)
            
        if progressive_thr:
            this_max, this_mean, this_std = make_stim2D_threshold(this_inverse, fwhm, width, mask)
        else:
            this_max = np.nanmax(this_inverse)
            this_mean = np.nanmean(this_inverse)
            this_std = np.nanstd(this_inverse)
            
            
        y, x = frames.shape
        twopi = 2 * np.pi
        yy = np.zeros((len(an_dist), n_fc))
        xx = np.zeros((len(an_dist), n_fc))
        fluxes = np.zeros((len(an_dist), n_fc))
        for k,a in enumerate(an_dist):
            for b in range(n_fc):
                sigposy = y / 2 + np.sin(b / n_fc * twopi) * a
                sigposx = x / 2 + np.cos(b / n_fc * twopi) * a
                
                yy[k,b] = sigposy
                xx[k,b] = sigposx
                
            apertures = CircularAperture(np.array((xx[k], yy[k])).T, np.mean(fwhm) / 2)
            these_fluxes = aperture_photometry(frames, apertures)
            these_fluxes = np.array(these_fluxes["aperture_sum"])
            fluxes[k] = these_fluxes
        
        stim_threshold = []
        stim_threshold.append([this_max, np.mean(this_inverse[pxl_mask]), 
                               np.std(this_inverse[pxl_mask]), 1, fluxes])
        
        if sigma is not None:
            stim_threshold[0][3] = (this_mean + sigma * this_std)/this_max


    completeness_theta_map = np.ones((len(an_dist), n_fc, 3))

    # We crop the PSF and check if PSF has been normalized (so that flux in
    # 1*FWHM aperture = 1) and fix if needed
    new_psf_size = int(round(3 * fwhm_med))
    if new_psf_size % 2 == 0:
        new_psf_size += 1
    # Normalize psf
    if len(cube.shape) == 3:
        psf = normalize_psf(
            psf, fwhm=fwhm, verbose=False, size=min(new_psf_size, psf.shape[1])
            )
    else:
        nch = cube.shape[0]
        V = [normalize_psf(psf[i], fwhm[i], size=20, imlib='ndimage-fourier', force_odd = True, full_output = True) for i in range(0, nch, 1)]
        psf, _, _ = [], [], []
        for i in range(0, nch, 1):
            psf.append(V[i][0])
        psf = np.array(psf)
    
    #nbr_to_detect = int(round(completeness * n_fc))
    #max_missed = int(n_fc - nbr_to_detect)
    
    
    if transmission is None:
        transmission_factors = np.ones(len(an_dist))
    else:
        transmission_factors = np.interp(an_dist, transmission[0], transmission[1], right = 1, left = 0)
    

    for k in range(len(an_dist)):
        a = an_dist[k]

        if verbose:
            print("*** Calculating contrast at r = {} ***".format(a))

        err_msg = "Could not converge on a contrast level matching required "
        err_msg += "completeness within {} iterations. Tested level: {}. "
        err_msg += "Is there too much self-subtraction? Consider decreasing "
        err_msg += "ncomp if using PCA, or increasing minimum requested radius."
        
        stim_maps = np.zeros((len(an_dist), n_fc, y, x))
        
        for b in range(0, n_fc):
            level = ini_contrast[k]
            stop_thr = level/precision
            prev = 0
            this_bound = [0, 1]
            cond = this_bound[0] == 0 or this_bound[1] == 1
            this_stim_map = np.zeros_like(cube[0])
            
            ii = 0
            while ii < max_iter:
                if verbose and not cond:
                    print('Current contrast bounds: ', this_bound)
            
                res = np.zeros((n_fc,2))

                this_result = _stim_fc(a,an_dist,b,level, n_fc, cube, psf, angle_list, 
                        fwhm, algo, algo_dict, stim_threshold, through_thresh, 
                        mask, conv, starphot, transmission_factors[k])
                
                res[b] = this_result[0:2]
                this_stim_map = this_result[2]
                
                if res[b][0] <= 0:
                    this_bound[0] = level
                    if this_bound[0] == 1:
                        print('No contrast lower than 1 found for theta = ', b * 360/n_fc)
                        ii = max_iter
                        break
                    if this_bound[1] == 1:
                        prev = level
                        level *= 1.5
                        if level > 1:
                            level = 1
                        stop_thr = level*precision
                    else:
                        prev = level
                        level = np.mean(this_bound)
                        stop_thr = level*precision
                else:
                    this_bound[1] = level
                    stim_maps[k, b] = this_stim_map
                    if this_bound[0] == 0:
                        prev = level
                        level *= 0.75
                        stop_thr = level*precision
                    else:
                        prev = level
                        level = np.mean(this_bound)
                        stop_thr = level*precision
                             
                if np.abs(this_bound[1] - this_bound[0]) < stop_thr*2:
                    if verbose:
                        print('Precision reached for theta = ', b * 360/n_fc, ' : ', this_bound)
                    break
                ii += 1
                        
            completeness_theta_map[k, b] = [level, *this_bound]
            
            if ii >= max_iter:
                if this_bound[0] == 1:
                    print('Check that there is not too much self-subtraction')
                else:
                    print('Could not find contrast for the requested precision within \
                      the amount of iterations allocated') 

    return an_dist, completeness_theta_map, stim_maps


# TODO: Include the algo_class in the metrics tutorial !!
def completeness_map(
    cube,
    angle_list,
    psf,
    fwhm,
    algo,
    an_dist,
    ini_contrast,
    starphot=1,
    n_fc=20,
    snr_approximation=True,
    nproc=1,
    algo_dict={},
    verbose=True,
    algo_class=None,
):
    """
    Function allowing the computation of two dimensional (radius and
    completeness) contrast curves with any psf-subtraction algorithm provided by
    VIP. The code relies on the approach proposed by [DAH21b]_, itself inspired
    by the framework developped in [JEN18]_. It relies on the computation of
    the contrast associated to a completeness level achieved at a level defined
    as the first false positive in the original SNR map (brightest speckle
    observed in the empty map). The computation of the completeness level
    associated to a contrast is done via the sequential injection of multiple
    fake companions. The algorithm uses multiple interpolations to find the
    contrast associated to the selected completeness level (0.95 by default).
    The function allows the computation of three dimensional completeness maps,
    with contrasts computed for multiple completeness level, allowing the
    reconstruction of the contrast/completeness distribution for every
    considered angular separations. For more details see [DAH21b]_.

    Parameters
    ----------
    cube : 3d or 4d numpy ndarray
        The input cube, 3d (ADI data) or 4d array (IFS data), without fake
        companions.
    angle_list : 1d numpy ndarray
        Vector of derotation angles to align North up in your cube images.
    psf : 2d or 3d numpy ndarray
        Frame with the psf template for the fake companion(s).
        PSF must be centered in array. Normalization is done internally.
    fwhm: int or float or 1d array, optional
        The the Full Width Half Maximum in pixels. It can handle a different
        FWHM value for different wavelengths (IFS data).
    algo : callable or function
        The post-processing algorithm, e.g. vip_hci.pca.pca.
    an_dist: list or ndarray
        List of angular separations for which a contrast has to be estimated.
        Default is None which corresponds to a range of spanning between 2
        FWHM and half the size of the provided cube - PSF size //2 with a
        step of 5 pixels
    ini_contrast: list, 1d ndarray or None, optional
        Initial contrast for the range of angular separations included in
        `an_dist`. The number of initial contrasts should be equivalent to the
        number of angular separations. Default is None which corresponds to the
        5-sigma contrast_curve obtained with ``vip_hci.metrics.contrast_curve``.
    starphot : int or float or 1d array
        If int or float it corresponds to the aperture photometry of the
        non-coronagraphic PSF which we use to scale the contrast.
        Default is 1 which corresponds to the contrast expressed in ADU.
    n_fc: int, optional
        Number of azimuths considered for the computation of the True
        positive rate/completeness, (number of fake companions injected
        separately). The range of achievable completeness depends on the
        number of considered azimuths (the minimum completeness is defined
        as 1/n_fc and the maximum is 1-1/n_fc). Default is 20.
    snr_approximated : bool, optional
        If True, an approximated S/N map is generated. If False the
        approach of [MAW14]_ is used. Default is True
    nproc : int or None
        Number of processes for parallel computing.
    algo_dict
        Any other valid parameter of the post-processing algorithms can be
        passed here, including e.g. imlib and interpolation.
    verbose : Boolean, optional
        If True the function prints intermediate info about the comptation of
        the completeness map. Default is True.

    Returns
    -------
    an_dist: 1d numpy ndarray
        Radial distances where the contrasts are calculated
    comp_levels: 1d numpy ndarray
        Completeness levels for which the contrasts are calculated
    cont_curve: 2d numpy ndarray
        Contrast matrix, with the first axis associated to the radial distances
        and the second axis associated to the completeness level, calculated
        from 1/n_fc to (n_fc-1)/n_fc.

    """

    if cube.ndim != 3 and cube.ndim != 4:
        raise TypeError("The input array is not a 3d or 4d cube")
    if cube.ndim == 3 and (cube.shape[0] != angle_list.shape[0]):
        raise TypeError("Input parallactic angles vector has wrong length")
    if cube.ndim == 4 and (cube.shape[1] != angle_list.shape[0]):
        raise TypeError("Input parallactic angles vector has wrong length")
    if cube.ndim == 3 and psf.ndim != 2:
        raise TypeError("Template PSF is not a frame (for ADI case)")
    if cube.ndim == 4 and psf.ndim != 3:
        raise TypeError("Template PSF is not a cube (for ADI+IFS case)")
    if nproc is None:
        nproc = cpu_count() // 2

    if isinstance(fwhm, (np.ndarray, list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if an_dist is None:
        an_dist = np.array(
            range(2 * round(fwhm), cube.shape[-1] // 2 - 2 * fwhm_med, 5)
        )
    elif an_dist[-1] > cube.shape[-1] // 2 - 2 * fwhm_med:
        raise TypeError("Please decrease the maximum annular distance")

    if ini_contrast is None:
        print("Contrast curve not provided => will be computed first...")
        # pxscale unused if plot=False
        ini_cc = contrast_curve(
            cube,
            angle_list,
            psf,
            fwhm_med,
            pxscale=0.1,
            starphot=starphot,
            algo=algo,
            sigma=3,
            plot=False,
            **algo_dict,
        )
        ini_rads = np.array(ini_cc["distance"])
        ini_cc = np.array(ini_cc["sensitivity_student"])

        if np.amax(an_dist) > np.amax(ini_rads):
            msg = "Max requested annular distance larger than covered by "
            msg += "contrast curve. Please decrease the maximum annular distance"
            raise ValueError(msg)

        # find closest contrast values to requested radii
        ini_contrast = []
        for aa, ad in enumerate(an_dist):
            idx = find_nearest(ini_rads, ad)
            ini_contrast.append(ini_cc[idx])

    # argl = getfullargspec(algo).args # does not work with recent object support

    """Because all psfsub algorithms now take more vague parameters, looking for
    specific named arguments can be a puzzle. We have to first identify the parameters
    tied to the algorithm by looking at its object of parameters."""

    # TODO: Clean below?
    # Consider 3 cases depending on whether algo is (i) defined externally,
    # (ii) a VIP postproc algorithm; (iii) ineligible for contrast curves
    argl = getfullargspec(algo).args
    if "cube" in argl and "angle_list" in argl and "verbose" in argl:
        # (i) external algorithm with appropriate parameters [OK]
        pass
    else:
        algo_name = algo.__name__
        idx = algo.__module__.index(
            '.', algo.__module__.index('.') + 1)
        mod = algo.__module__[:idx]
        tmp = __import__(
            mod, fromlist=[algo_name.upper()+'_Params'])
        algo_params = getattr(tmp, algo_name.upper()+'_Params')
        argl = [attr for attr in vars(algo_params)]
        if "cube" in argl and "angle_list" in argl and "verbose" in argl:
            # (ii) a VIP postproc algorithm [OK]
            pass
        else:
            # (iii) ineligible routine for contrast curves [Raise error]
            msg = "Ineligible algo for contrast curve function. algo should "
            msg += "have parameters 'cube', 'angle_list' and 'verbose'"
            raise TypeError(msg)

    if "cube" in argl and "angle_list" in argl and "verbose" in argl:
        if "fwhm" in argl:
            frame_fin = algo(
                cube=cube,
                angle_list=angle_list,
                fwhm=fwhm_med,
                verbose=False,
                **algo_dict,
            )
        else:
            frame_fin = algo(
                cube=cube, angle_list=angle_list, verbose=False, **algo_dict
            )

    snrmap_empty = snrmap(
        frame_fin,
        fwhm_med,
        approximated=snr_approximation,
        plot=False,
        known_sources=None,
        nproc=nproc,
        array2=None,
        use2alone=False,
        exclude_negative_lobes=False,
        verbose=False,
    )

    contrast_matrix = np.zeros((len(an_dist), n_fc + 1))
    detect_pos_matrix = [[]] * (n_fc + 1)

    for k in range(0, len(an_dist)):
        a = an_dist[k]
        level = ini_contrast[k]
        pos_detect = []
        det_bound = [None, None]
        lvl_bound = [None, None]

        print("Starting annulus " + "{}".format(a))

        while len(pos_detect) == 0:
            pos_detect = []
            pos_non_detect = []
            res = pool_map(
                nproc,
                _estimate_snr_fc,
                a,
                iterable(range(0, n_fc)),
                level,
                n_fc,
                cube,
                psf,
                angle_list,
                fwhm,
                algo,
                algo_dict,
                snrmap_empty,
                starphot,
                approximated=snr_approximation,
            )

            for res_i in res:
                if res_i[0] > 0:
                    pos_detect.append(res_i[1])
                else:
                    pos_non_detect.append(res_i[1])

            contrast_matrix[k, len(pos_detect)] = level
            detect_pos_matrix[len(pos_detect)] = [
                list(pos_detect.copy()),
                list(pos_non_detect.copy()),
            ]
            if len(pos_detect) == 0:
                level = level * 1.5

        while contrast_matrix[k, 0] == 0:
            level = level * 0.75
            res = pool_map(
                nproc,
                _estimate_snr_fc,
                a,
                iterable(-np.sort(-np.array(pos_detect))),
                level,
                n_fc,
                cube,
                psf,
                angle_list,
                fwhm,
                algo,
                algo_dict,
                snrmap_empty,
                starphot,
                approximated=snr_approximation,
            )

            it = len(pos_detect) - 1
            for res_i in res:
                if res_i[0] < 0:
                    pos_non_detect.append(res_i[1])
                    del pos_detect[it]
                it -= 1

            contrast_matrix[k, len(pos_detect)] = level
            detect_pos_matrix[len(pos_detect)] = [
                list(pos_detect.copy()),
                list(pos_non_detect.copy()),
            ]

        if verbose:
            print("Lower bound ({:.0f}%) found: {}".format(100 / n_fc, level))

        level = contrast_matrix[k, np.where(contrast_matrix[k, :] > 0)[0][-1]]

        pos_detect = []
        pos_non_detect = list(np.arange(0, n_fc))

        while contrast_matrix[k, n_fc] == 0:
            level = level * 1.25

            res = pool_map(
                nproc,
                _estimate_snr_fc,
                a,
                iterable(-np.sort(-np.array(pos_non_detect))),
                level,
                n_fc,
                cube,
                psf,
                angle_list,
                fwhm,
                algo,
                algo_dict,
                snrmap_empty,
                starphot,
                approximated=snr_approximation,
            )

            it = len(pos_non_detect) - 1
            for res_i in res:
                if res_i[0] > 0:
                    pos_detect.append(res_i[1])
                    del pos_non_detect[it]
                it -= 1

            contrast_matrix[k, len(pos_detect)] = level
            detect_pos_matrix[len(pos_detect)] = [
                list(pos_detect.copy()),
                list(pos_non_detect.copy()),
            ]

        if verbose:
            print(
                "Upper bound ({:.0f}%) found: {}".format(
                    100 * (n_fc - 1) / n_fc, level)
            )

        missing = np.where(contrast_matrix[k, :] == 0)[0]
        computed = np.where(contrast_matrix[k, :] > 0)[0]
        while len(missing) > 0:
            pos_temp = np.argmax((computed - missing[0])[computed < missing[0]])
            det_bound[0] = computed[pos_temp]
            lvl_bound[0] = contrast_matrix[k, det_bound[0]]
            sort_temp = np.sort((missing[0] - computed))
            sort_temp = sort_temp[np.sort((missing[0] - computed)) < 0]
            det_bound[1] = -np.sort(-computed)[np.argmax(sort_temp)]
            lvl_bound[1] = contrast_matrix[k, det_bound[1]]
            it = 0
            while len(pos_detect) != missing[0]:
                if (
                    np.argmin(
                        [
                            len(detect_pos_matrix[det_bound[1]][0]),
                            len(detect_pos_matrix[det_bound[0]][1]),
                        ]
                    )
                    == 0
                ):
                    pos_detect = np.sort(detect_pos_matrix[det_bound[1]][0])
                    pos_non_detect = np.sort(detect_pos_matrix[det_bound[1]][1])
                    pos_detect = list(pos_detect)
                    pos_non_detect = list(pos_non_detect)
                    num = lvl_bound[1] - lvl_bound[0]
                    denom = det_bound[1] - det_bound[0]
                    level = lvl_bound[1] + num * \
                        (missing[0] - det_bound[1]) / denom

                    res = pool_map(
                        nproc,
                        _estimate_snr_fc,
                        a,
                        iterable(-np.sort(-np.array(pos_detect))),
                        level,
                        n_fc,
                        cube,
                        psf,
                        angle_list,
                        fwhm,
                        algo,
                        algo_dict,
                        snrmap_empty,
                        starphot,
                        approximated=snr_approximation,
                    )

                    it = len(pos_detect) - 1
                    for res_i in res:
                        if res_i[0] < 0:
                            pos_non_detect.append(res_i[1])
                            del pos_detect[it]
                        it -= 1

                else:
                    pos_detect = np.sort(detect_pos_matrix[det_bound[0]][0])
                    pos_non_detect = np.sort(detect_pos_matrix[det_bound[0]][1])
                    pos_detect = list(pos_detect)
                    pos_non_detect = list(pos_non_detect)
                    num = lvl_bound[1] - lvl_bound[0]
                    denom = det_bound[1] - det_bound[0]
                    level = lvl_bound[0] + num * \
                        (missing[0] - det_bound[0]) / denom

                    res = pool_map(
                        nproc,
                        _estimate_snr_fc,
                        a,
                        iterable(-np.sort(-np.array(pos_non_detect))),
                        level,
                        n_fc,
                        cube,
                        psf,
                        angle_list,
                        fwhm,
                        algo,
                        algo_dict,
                        snrmap_empty,
                        starphot,
                        approximated=snr_approximation,
                    )

                    it = len(pos_non_detect) - 1
                    for res_i in res:
                        if res_i[0] > 0:
                            pos_detect.append(res_i[1])
                            del pos_non_detect[it]
                        it -= 1

                if len(pos_detect) > missing[0]:
                    det_bound[1] = len(pos_detect)
                    lvl_bound[1] = level
                elif len(pos_detect) < missing[0] and lvl_bound[0] < level:
                    det_bound[0] = len(pos_detect)
                    lvl_bound[0] = level

                contrast_matrix[k, len(pos_detect)] = level
                detect_pos_matrix[len(pos_detect)] = [
                    list(pos_detect.copy()),
                    list(pos_non_detect.copy()),
                ]

                if len(pos_detect) == missing[0]:
                    if verbose:
                        print(
                            "Data point "
                            + "{}".format(len(pos_detect) / n_fc)
                            + " found. Still "
                            + "{}".format(len(missing) - it - 1)
                            + " data point(s) missing"
                        )

            computed = np.where(contrast_matrix[k, :] > 0)[0]
            missing = np.where(contrast_matrix[k, :] == 0)[0]

    comp_levels = np.linspace(1 / n_fc, 1 - 1 / n_fc, n_fc - 1, endpoint=True)

    return an_dist, comp_levels, contrast_matrix[:, 1:-1]
