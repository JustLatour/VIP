# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 12:16:20 2023

@author: justin
"""

import numpy as np
from multiprocessing import cpu_count
from hciplot import plot_frames, plot_cubes
from typing import Tuple, Union, List
from dataclasses import dataclass
from enum import Enum
import numbers
from photutils.aperture import CircularAperture
from .svd import svd_wrapper, SVDecomposer
from .utils_pca import pca_incremental, pca_grid
from ..config import (timing, time_ini, check_enough_memory, Progressbar,
                      check_array)
from ..config.paramenum import (SvdMode, Adimsdi, Interpolation, Imlib, Collapse,
                                ALGO_KEY)
from ..config.utils_conf import pool_map, iterable
from ..config.utils_param import setup_parameters, separate_kwargs_dict
from ..preproc.derotation import _find_indices_adi, _find_indices_adi2, _compute_pa_thresh, _define_annuli
from ..preproc import cube_rescaling_wavelengths as scwave
from ..preproc import (cube_derotate, cube_collapse, cube_subtract_sky_pca,
                       check_pa_vector, check_scal_vector, cube_crop_frames)
from ..stats import descriptive_stats
from ..var import (frame_center, dist, prepare_matrix, reshape_matrix,
                   cube_filter_lowpass, mask_circle, get_annulus_segments)


from .pca_fullfr import *
from .pca_fullfr import PCA_Params
from .pca_local import *
from .pca_local import PCA_ANNULAR_Params


@dataclass
class PCA_LOCAL_MASK_Params(PCA_Params):
    """
    Set of parameters for the multi-epoch pca
    """
    cube: np.ndarray = None
    angle_list: np.ndarray = None
    cube_ref: np.ndarray = None
    scale_list: np.ndarray = None
    radius_int: int = 0
    fwhm: float = 4
    asize: float = 6
    n_segments: int = 8
    location: Union["all", float] = "all"
    mask_rdi: np.ndarray = None
    delta_rot: Union[float, Tuple[float]] = (0.1, 1)
    ncomp: Union[int, Tuple, np.ndarray] = 1
    svd_mode: Enum = SvdMode.LAPACK
    nproc: int = 1
    min_frames_lib: int = 10
    max_frames_lib: int = 200
    tol: float = 1e-1
    scaling: Enum = None
    imlib: Enum = Imlib.VIPFFT
    interpolation: Enum = Interpolation.LANCZOS4
    collapse: Enum = Collapse.MEDIAN
    collapse_ifs: Enum = Collapse.MEAN
    ifs_collapse_range: Union["all", Tuple[int]] = "all"
    theta_init: int = 0
    weights: np.ndarray = None
    cube_sig: np.ndarray = None
    full_output: bool = False
    verbose: bool = True
    left_eigv: bool = False
    
    
def pca_local_mask(*all_args: List, **all_kwargs: dict):
    
    class_params, rot_options = separate_kwargs_dict(initial_kwargs=all_kwargs,
                                parent_class=PCA_LOCAL_MASK_Params)
    
    algo_params = PCA_LOCAL_MASK_Params(*all_args, **class_params)
    
    if algo_params.radius_int and len(rot_options) == 0:
        rot_options['mask_val'] = 0
        rot_options['ker'] = 1
        rot_options['interp_zeros'] = True

    global start_time
    start_time = time_ini()
    
    
    
    
    
    #???
    if algo_params.cube.ndim == 3:
        add_params = {"start_time": start_time, "full_output": True}
        
        n, y, x = cube_shape
        NbrImages = n
        if (algo_params.step_corr == 1 and NbrImages == algo_params.cube.shape[0]
            and algo_params.ADI_Lib is None and algo_params.RDI_Lib is None):
            func_params = setup_parameters(
                params_obj=algo_params, fkt=_pca_adi_rdi, **add_params
            )
            if func_params['max_frames_lib'] is None:
                func_params['max_frames_lib'] = 200
            if func_params['min_frames_lib'] is None:
                func_params['min_frames_lib'] = 10
            res = _pca_adi_rdi(**func_params, **rot_options)
        else:
            func_params = setup_parameters(
                params_obj=algo_params, fkt=_pca_adi_rdi_corr, **add_params
            )
            res = _pca_adi_rdi_corr(**func_params, **rot_options)

        cube_out, cube_der, frame = res
        if algo_params.full_output:
            return cube_out, cube_der, frame
        else:
            return frame
    
    pass


def pca_annular_mask(
    cube,
    angle_list,
    radius_int,
    fwhm,
    asize,
    n_annuli = 'auto',
    n_segments = 8,
    mask_rdi = None,
    delta_rot=0,
    ncomp=1,
    svd_mode="lapack",       
    nproc=None,
    min_frames_lib=2,
    max_frames_lib=200,
    tol=1e-1,
    scaling=None,
    imlib="vip-fft",
    interpolation="lanczos4",
    collapse="median",
    full_output=False,
    verbose=1,
    cube_ref=None,
    theta_init=0,
    weights=None,
    cube_sig=None,
    left_eigv=False,
    **rot_options
 ):
    array = cube
    if array.ndim != 3:
        raise TypeError("Input array is not a cube or 3d array")
    if array.shape[0] != angle_list.shape[0]:
        raise TypeError("Input vector or parallactic angles has wrong length")

    n, y, x = array.shape
    
    global start_time
    start_time = time_ini()

    angle_list = check_pa_vector(angle_list)
    n_annuli = int((y / 2 - radius_int) / asize)
    
    if isinstance(n_segments, list):
        if len(n_segments) != n_annuli:
            raise ValueError('If n_segments is a list, its length must be the same as the number of annuli')
    elif np.isscalar(n_segments):
        n_segments = [n_segments] * n_annuli
    
    if isinstance(delta_rot, tuple):
        delta_rot = np.linspace(delta_rot[0], delta_rot[1], num=n_annuli)
    elif np.isscalar(delta_rot):
        delta_rot = [delta_rot] * n_annuli

    # if isinstance(n_segments, int):
    #     n_segments = [n_segments for _ in range(n_annuli)]
    # elif n_segments == "auto":
    #     n_segments = list()
    #     n_segments.append(2)  # for first annulus
    #     n_segments.append(3)  # for second annulus
    #     ld = 2 * np.tan(360 / 4 / 2) * asize
    #     for i in range(2, n_annuli):  # rest of annuli
    #         radius = i * asize
    #         ang = np.rad2deg(2 * np.arctan(ld / (2 * radius)))
    #         n_segments.append(int(np.ceil(360 / ang)))

    if verbose:
        msg = "N annuli = {}, FWHM = {:.3f}"
        print(msg.format(n_annuli, fwhm))
        print("PCA per annulus (or annular sectors):")

    if nproc is None:  # Hyper-threading "duplicates" the cores -> cpu_count/2
        nproc = cpu_count() // 2
        
    if isinstance(ncomp, list):
        nncomp = len(ncomp)
        cube_out = np.zeros([nncomp, array.shape[0], array.shape[1],
                             array.shape[2]])
    elif isinstance(ncomp, np.ndarray):
        #1st dim of ncomp is all the epochs
        #2nd dim is the ncomp values tested
        nncomp = ncomp.shape[1]
        cube_out = np.zeros([nncomp, array.shape[0], array.shape[1],
                             array.shape[2]])
    elif np.isscalar(ncomp):
        nncomp = 1
        cube_out = np.zeros([1, array.shape[0], array.shape[1],
                             array.shape[2]])
    for ann in range(n_annuli):
        if isinstance(ncomp, tuple):
            if len(ncomp) == n_annuli:
                ncompann = ncomp[ann]
            else:
                msg = "If `ncomp` is a tuple, its length must match the number "
                msg += "of annuli"
                raise TypeError(msg)
        else:
            ncompann = ncomp

        res_ann_par = _define_annuli(
            angle_list,
            ann,
            n_annuli,
            fwhm,
            radius_int,
            asize,
            delta_rot[ann],
            1,
            verbose,
            True,
        )
        pa_thr, inner_radius, ann_center = res_ann_par
        
        cube_out += pca_ardi_annulus_mask(
            cube,
            angle_list,
            inner_radius,
            fwhm,
            asize,
            n_segments[ann],
            mask_rdi,
            pa_thr,
            ncompann,
            svd_mode,
            nproc,
            min_frames_lib,
            max_frames_lib,
            tol,
            scaling,
            imlib,
            interpolation,
            collapse,
            -1,
            verbose,
            cube_ref,
            theta_init,
            weights,
            cube_sig,
            left_eigv,
            **rot_options
        )
        
    result = np.zeros((nncomp,y,x))
    cube_der = np.zeros_like(cube_out)
    for n in range(nncomp):
        cube_der[n,:,:,:] = cube_derotate(
            cube_out[n],
            angle_list,
            nproc=nproc,
            imlib=imlib,
            interpolation=interpolation,
            **rot_options,
        )
        result[n,:,:] = cube_collapse(cube_der[n,:,:,:], mode=collapse, w=weights)
        
    if verbose:
        print("Done derotating and combining.")
        timing(start_time)
        
    if nncomp == 1:
        result = result[0,:,:]
        
    if full_output:
        return cube_out, cube_der, result
    else:
        return result
    
    
def pca_annular_masked(
    cube,
    angle_list,
    radius_int,
    fwhm,
    asize,
    n_annuli = 'auto',
    n_segments = 8,
    mask_rdi = None,
    delta_rot=0,
    ncomp=1,
    svd_mode="lapack",       
    nproc=None,
    min_frames_lib=2,
    max_frames_lib=200,
    tol=1e-1,
    scaling=None,
    imlib="vip-fft",
    interpolation="lanczos4",
    collapse="median",
    full_output=False,
    verbose=1,
    cube_ref=None,
    theta_init=0,
    weights=None,
    cube_sig=None,
    left_eigv=False,
    **rot_options
 ):
    array = cube
    if array.ndim != 3:
        raise TypeError("Input array is not a cube or 3d array")
    if array.shape[0] != angle_list.shape[0]:
        raise TypeError("Input vector or parallactic angles has wrong length")

    n, y, x = array.shape
    
    global start_time
    start_time = time_ini()

    angle_list = check_pa_vector(angle_list)
    n_annuli = int((y / 2 - radius_int) / asize)
    
    if isinstance(n_segments, list):
        if len(n_segments) != n_annuli:
            raise ValueError('If n_segments is a list, its length must be the same as the number of annuli')
    elif np.isscalar(n_segments):
        n_segments = [n_segments] * n_annuli
    
    if isinstance(delta_rot, tuple):
        delta_rot = np.linspace(delta_rot[0], delta_rot[1], num=n_annuli)
    elif np.isscalar(delta_rot):
        delta_rot = [delta_rot] * n_annuli

    # if isinstance(n_segments, int):
    #     n_segments = [n_segments for _ in range(n_annuli)]
    # elif n_segments == "auto":
    #     n_segments = list()
    #     n_segments.append(2)  # for first annulus
    #     n_segments.append(3)  # for second annulus
    #     ld = 2 * np.tan(360 / 4 / 2) * asize
    #     for i in range(2, n_annuli):  # rest of annuli
    #         radius = i * asize
    #         ang = np.rad2deg(2 * np.arctan(ld / (2 * radius)))
    #         n_segments.append(int(np.ceil(360 / ang)))

    if verbose:
        msg = "N annuli = {}, FWHM = {:.3f}"
        print(msg.format(n_annuli, fwhm))
        print("PCA per annulus (or annular sectors):")

    if nproc is None:  # Hyper-threading "duplicates" the cores -> cpu_count/2
        nproc = cpu_count() // 2
        
    if isinstance(ncomp, list):
        nncomp = len(ncomp)
        cube_out = np.zeros([nncomp, array.shape[0], array.shape[1],
                             array.shape[2]])
    elif np.isscalar(ncomp):
        nncomp = 1
        cube_out = np.zeros([1, array.shape[0], array.shape[1],
                             array.shape[2]])
    for ann in range(n_annuli):
        if isinstance(ncomp, tuple) or isinstance(ncomp, np.ndarray):
            if len(ncomp) == n_annuli:
                ncompann = ncomp[ann]
            else:
                msg = "If `ncomp` is a tuple, its length must match the number "
                msg += "of annuli"
                raise TypeError(msg)
        else:
            ncompann = ncomp

        res_ann_par = _define_annuli(
            angle_list,
            ann,
            n_annuli,
            fwhm,
            radius_int,
            asize,
            delta_rot[ann],
            1,
            verbose,
            True,
        )
        pa_thr, inner_radius, ann_center = res_ann_par
        
        cube_out += pca_ardi_annulus_masked(
            cube,
            angle_list,
            inner_radius,
            fwhm,
            asize,
            n_segments[ann],
            mask_rdi,
            pa_thr,
            ncompann,
            svd_mode,
            nproc,
            min_frames_lib,
            max_frames_lib,
            tol,
            scaling,
            imlib,
            interpolation,
            collapse,
            True,
            verbose,
            cube_ref,
            theta_init,
            weights,
            cube_sig,
            left_eigv,
            **rot_options
        )
        
    result = np.zeros((nncomp,y,x))
    cube_der = np.zeros_like(cube_out)
    for n in range(nncomp):
        cube_der[n,:,:,:] = cube_derotate(
            cube_out[n],
            angle_list,
            nproc=nproc,
            imlib=imlib,
            interpolation=interpolation,
            **rot_options,
        )
        result[n,:,:] = cube_collapse(cube_der[n,:,:,:], mode=collapse, w=weights)
        
    if verbose:
        print("Done derotating and combining.")
        timing(start_time)
        
    if full_output:
        return cube_out, cube_der, result
    else:
        return result


def pca_ardi_annulus_mask(
    cube,
    angle_list,
    inner_radius,
    fwhm,
    asize,
    n_segments = 8,
    mask_rdi = None,
    pa_thr=0,
    ncomp=1,
    svd_mode="lapack",       
    nproc=None,
    min_frames_lib=2,
    max_frames_lib=200,
    tol=1e-1,
    scaling=None,
    imlib="vip-fft",
    interpolation="lanczos4",
    collapse="median",
    full_output=False,
    verbose=1,
    cube_ref=None,
    theta_init=0,
    weights=None,
    cube_sig=None,
    left_eigv=False,
    **rot_options,
):
    """
    Small boat following the targeted zone
    
    asize and inner_radius in pixels
    location in degrees
    step_location in degrees
    -mask_rdi: location on which project the components. anchor. 
        If None, full-frame
        If mask_rdi == 'annulus', only the rest of the annulus
    """
    array = cube
    if array.ndim != 3:
        raise TypeError("Input array is not a cube or 3d array")
    if array.shape[0] != angle_list.shape[0]:
        raise TypeError("Input vector or parallactic angles has wrong length")
        
        
    if verbose:
        print("PCA per annulus (or annular sectors):")

    n, y, x = array.shape

    angle_list = check_pa_vector(angle_list)
    if np.isscalar(ncomp):
        ncomp = np.array([ncomp])
    if isinstance(ncomp, list):
        ncomp = np.array(ncomp)
    
    angular_range = 360 #in degrees
    ann_center = inner_radius+(asize/2)

    angular_width = angular_range/n_segments
    centers = [theta_init + angular_width/2]
    total_rot = 0
    while (total_rot+angular_width) < 360:
        total_rot += angular_width
        centers.append(centers[-1]+angular_width)

    n_images = len(centers)
    
    yy,xx = get_annulus_segments(cube[0], inner_radius, asize, 1, 0)[0]
    mask_annulus = np.zeros_like(cube[0], dtype = int)
    mask_annulus[yy,xx] = 1
    
    yy, xx = np.where(mask_annulus == 1)
    
    #then convert to bool, boucle on centers, get mask of zone to subtract, rotate it per image...
    cy, cx = frame_center(cube[0])
    twopi = 2*np.pi
    
    masks_centers = []
    
    nnpcs = ncomp.shape[0]
    if len(ncomp.shape) == 2:
        nnpcs = ncomp.shape[1]
    
    results = np.zeros((n_images, nnpcs, cube.shape[1], cube.shape[2]))
    result_noder = np.zeros((nnpcs,n,cube.shape[1],cube.shape[2]))
    
    nbr_frames = []
    
    Indices_segments = get_annulus_segments(cube[0], inner_radius, asize, n_segments, theta_init)
    for i, center in enumerate(centers):
        boat = np.zeros_like(cube[0])
        boat[Indices_segments[i][0], Indices_segments[i][1]] = 1
        
        #plot_frames(boat)
        
        sci_cube_skysub = np.zeros((nnpcs, cube.shape[0], cube.shape[1], cube.shape[2]))
        for k in range(cube.shape[0]):
            position = center - angle_list[k]
            boat_k = mask_local_boat(mask_annulus, position, angular_width)
            if mask_rdi is None:
                anchor_k = np.ones_like(cube[0]) - boat_k
            elif mask_rdi == 'annulus':
                anchor_k = mask_annulus - boat_k
            
            #plot_frames(boat_k)
            
            adi_indices = _find_indices_adi2(angle_list, k, pa_thr,
                                             truncate=True,
                                             max_frames=max_frames_lib)
            adi_indices = np.sort(adi_indices)
            
            frame_boat = cube[k] * boat_k
            frame_anchor = cube[k] * anchor_k
            
            frame_anchor_l = prepare_matrix(frame_anchor[np.newaxis,:,:], scaling=None,
                                                verbose=False)
            
            cube_used = cube[adi_indices]
            if cube_ref is not None:
                cube_used = np.vstack((cube_used, cube_ref))
                
            nbr_frames.append(cube_used.shape[0])
            
            cube_anchor = cube_used * anchor_k
            cube_anchor_l = prepare_matrix(cube_anchor, scaling=None, verbose=False)
        
            cube_boat = cube_used * boat_k
            cube_boat_l = prepare_matrix(cube_boat, scaling=None, verbose=False)
        
            sky_kl = np.dot(cube_anchor_l, cube_anchor_l.T)
            #print(sky_kl)
            #print(sky_kl.shape)
            Msky_kl = prepare_matrix(sky_kl, scaling=None, verbose=False)
            sky_pcs = svd_wrapper(Msky_kl, 'lapack', sky_kl.shape[0], False)
            sky_pcs_kl = sky_pcs.reshape(sky_kl.shape[0], sky_kl.shape[1])
        
            sky_pc_anchor = np.dot(sky_pcs_kl, cube_anchor_l)
            sky_pcs_anchor_cube = sky_pc_anchor.reshape(cube_anchor.shape[0],
                                                    cube_anchor.shape[1],
                                                    cube_anchor.shape[2])
        
            sky_pcs_boat_cube = np.dot(sky_pcs_kl, cube_boat_l).reshape(cube_anchor.shape[0],
                                                                 cube_anchor.shape[1],
                                                                 cube_anchor.shape[2])
            #print(sky_pcs_boat_cube.shape)
            transf_sci = np.zeros((cube_anchor.shape[0]))  #sky number image, science number image
           
            transf_sci = np.inner(sky_pc_anchor, frame_anchor_l[0].T)

            #print(transf_sci.shape)
            #print(transf_sci)
            Msky_pcs_anchor = prepare_matrix(sky_pcs_anchor_cube, scaling=None,
                                         verbose=False)

            mat_inv = np.linalg.inv(np.dot(Msky_pcs_anchor, Msky_pcs_anchor.T))
            transf_sci_scaled = np.dot(mat_inv, transf_sci)
            #print(transf_sci_scaled)
        
            tmp_sky = np.zeros_like(cube[0])
            if len(ncomp.shape) == 2:
                for n in range(np.max(ncomp[k])):
                    tmp_sky += np.array([transf_sci_scaled[n]*sky_pcs_boat_cube[n]]).reshape(cube.shape[1], cube.shape[2])
                    if n+1 in ncomp[k]:
                        index = np.where(ncomp[k] == n+1)[0][0]
                        sci_cube_skysub[index,k] = frame_boat - tmp_sky
            else:
                for n in range(np.max(ncomp)):
                    tmp_sky += np.array([transf_sci_scaled[n]*sky_pcs_boat_cube[n]]).reshape(cube.shape[1], cube.shape[2])
                    if n+1 in ncomp:
                        index = np.where(ncomp == n+1)[0][0]
                        sci_cube_skysub[index,k] = frame_boat - tmp_sky
                   
            result_noder[:,k,:,:] += sci_cube_skysub[:,k] * boat_k
            
        if verbose:
            print("segment {} done".format(i))
                    
    #derotation
    if full_output != -1:
        cube_der = np.zeros_like(sci_cube_skysub)
        for n in range(nnpcs):
            cube_der[n,:,:,:] = cube_derotate(
                result_noder[n,:,:,:],
                angle_list,
                nproc=nproc,
                imlib=imlib,
                interpolation=interpolation,
                **rot_options,
            )
            results[i,n,:,:] = cube_collapse(cube_der[n,:,:,:], mode=collapse, w=weights)
            results[i,n,:,:] *= mask_annulus
        
        
    results = np.sum(results, axis = 0)
    
    if verbose:
        print(np.mean(nbr_frames))

    if len(ncomp) == 1:
        results = results[0,:,:]
    #ncomp is a list or not??
    if full_output == -1:
        return result_noder
    elif full_output == True:
        return result_noder, cube_der, results
    else:
        return results


def pca_ardi_annulus_masked(
    cube,
    angle_list,
    inner_radius,
    fwhm,
    asize,
    n_segments = 8,
    mask_rdi = None,
    pa_thr=0,
    ncomp=1,
    svd_mode="lapack",       
    nproc=None,
    min_frames_lib=2,
    max_frames_lib=200,
    tol=1e-1,
    scaling=None,
    imlib="vip-fft",
    interpolation="lanczos4",
    collapse="median",
    full_output=False,
    verbose=1,
    cube_ref=None,
    theta_init=0,
    weights=None,
    cube_sig=None,
    left_eigv=False,
    **rot_options,
):
    """
    Larger boat covering the whole movement of the parallactic angle
    
    asize and inner_radius in pixels
    location in degrees
    step_location in degrees
    -mask_rdi: location on which project the components. anchor. 
        If None, full-frame
        If mask_rdi == 'annulus', only the rest of the annulus
    """
    array = cube
    if array.ndim != 3:
        raise TypeError("Input array is not a cube or 3d array")
    if array.shape[0] != angle_list.shape[0]:
        raise TypeError("Input vector or parallactic angles has wrong length")
        
        
    if verbose:
        print("PCA per annulus (or annular sectors):")
        
    global start_time
    start_time = time_ini()

    n, y, x = array.shape

    angle_list = check_pa_vector(angle_list)
    if np.isscalar(ncomp):
        ncomp = np.array([ncomp])
    if isinstance(ncomp, list):
        ncomp = np.array(ncomp)
    
    angular_range = 360 #in degrees
    ann_center = inner_radius+(asize/2)


    para_range = np.max(angle_list) - np.min(angle_list)
    angular_width = angular_range/n_segments
    centers = [theta_init + angular_width/2]
    total_rot = 0
    while (total_rot+angular_width) < 360:
        total_rot += angular_width
        centers.append(centers[-1]+angular_width)

    n_images = len(centers)
    
    yy,xx = get_annulus_segments(cube[0], inner_radius, asize, 1, 0)[0]
    mask_annulus = np.zeros_like(cube[0], dtype = int)
    mask_annulus[yy,xx] = 1
    
    yy, xx = np.where(mask_annulus == 1)
    
    #then convert to bool, boucle on centers, get mask of zone to subtract, rotate it per image...
    cy, cx = frame_center(cube[0])
    
    masks_centers = []
    
    nnpcs = ncomp.shape[0]
    
    results = np.zeros((n_images, nnpcs, cube.shape[1], cube.shape[2]))
    result_noder = np.zeros((nnpcs,n,cube.shape[1],cube.shape[2]))
    
    Indices_segments = get_annulus_segments(cube[0], inner_radius, asize, n_segments, theta_init)
    for i, center in enumerate(centers):
        boat = np.zeros_like(cube[0])
        boat[Indices_segments[i][0], Indices_segments[i][1]] = 1
        
        sci_cube_skysub = np.zeros((nnpcs, cube.shape[0], cube.shape[1], cube.shape[2]))
        
        position = center + para_range/2 - angle_list[-1]
        boat_k = mask_local_boat(mask_annulus, position, angular_width+para_range)
        if mask_rdi is None:
            anchor_k = np.ones_like(cube[0]) - boat_k
        elif mask_rdi == 'annulus':
            anchor_k = mask_annulus - boat_k
        
        #boat_k = np.ones_like(cube[0])
        
        if pa_thr == 0 and 1 == 3:
            cube_anchor = cube * anchor_k
            cube_anchor_l = prepare_matrix(cube_anchor, scaling=None, verbose=False)
            
            cube_boat = cube * boat_k
            cube_boat_l = prepare_matrix(cube_boat, scaling=None, verbose=False)
            
            sky_kl = np.dot(cube_anchor_l, cube_anchor_l.T)
            #print(sky_kl)
            #print(sky_kl.shape)
            Msky_kl = prepare_matrix(sky_kl, scaling=None, verbose=False)
            sky_pcs = svd_wrapper(Msky_kl, 'lapack', sky_kl.shape[0], False)
            sky_pcs_kl = sky_pcs.reshape(sky_kl.shape[0], sky_kl.shape[1])
            
            sky_pc_anchor = np.dot(sky_pcs_kl, cube_anchor_l)
            sky_pcs_anchor_cube = sky_pc_anchor.reshape(cube_anchor.shape[0],
                                                        cube_anchor.shape[1],
                                                        cube_anchor.shape[2])
            
            sky_pcs_boat_cube = np.dot(sky_pcs_kl, cube_boat_l).reshape(cube_anchor.shape[0],
                                                                     cube_anchor.shape[1],
                                                                     cube_anchor.shape[2])
            #print(sky_pcs_boat_cube.shape)
            transf_sci = np.zeros((cube_anchor.shape[0], cube_anchor.shape[0]))
            for j in range(cube_anchor.shape[0]):
                transf_sci[:, j] = np.inner(sky_pc_anchor, cube_anchor_l[j].T)

            #print(transf_sci.shape)
            #print(transf_sci)
            Msky_pcs_anchor = prepare_matrix(sky_pcs_anchor_cube, scaling=None,
                                             verbose=False)

            mat_inv = np.linalg.inv(np.dot(Msky_pcs_anchor, Msky_pcs_anchor.T))
            transf_sci_scaled = np.dot(mat_inv, transf_sci)
            #print(transf_sci_scaled)
            
            for k in range(cube.shape[0]):
                tmp_sky = np.zeros_like(cube[0])
                position = center - angle_list[k]
                boat_final = mask_local_boat(mask_annulus, position, angular_width)
                for n in range(np.max(ncomp)):
                    tmp_sky += np.array(transf_sci_scaled[n, k]*sky_pcs_boat_cube[n]).reshape(cube.shape[1], cube.shape[2])
                    if n+1 in ncomp:
                        index = np.where(ncomp == n+1)[0][0]
                        sci_cube_skysub[index,k] = cube_boat[k] - tmp_sky
                    
                result_noder[:,k,:,:] += sci_cube_skysub[:,k] * boat_final
                #plot_frames(result_noder[15,k,:,:])
                        
        else:
            for k in range(cube.shape[0]):
                adi_indices = _find_indices_adi2(angle_list, k, pa_thr,
                                                 truncate=True,
                                                 max_frames=max_frames_lib)
                adi_indices = np.sort(adi_indices)
                
                frame_boat = cube[k] * boat_k
                frame_anchor = cube[k] * anchor_k
                
                frame_anchor_l = prepare_matrix(frame_anchor[np.newaxis,:,:], scaling=None,
                                                    verbose=False)
                
                cube_used = cube[adi_indices]
                if cube_ref is not None:
                    cube_used = np.vstack((cube_used, cube_ref))
                
                
                cube_anchor = cube_used * anchor_k
                cube_anchor_l = prepare_matrix(cube_anchor, scaling=None, verbose=False)
            
                cube_boat = cube_used * boat_k
                cube_boat_l = prepare_matrix(cube_boat, scaling=None, verbose=False)
            
                sky_kl = np.dot(cube_anchor_l, cube_anchor_l.T)
                #print(sky_kl)
                #print(sky_kl.shape)
                Msky_kl = prepare_matrix(sky_kl, scaling=None, verbose=False)
                sky_pcs = svd_wrapper(Msky_kl, 'lapack', sky_kl.shape[0], False)
                sky_pcs_kl = sky_pcs.reshape(sky_kl.shape[0], sky_kl.shape[1])
            
                sky_pc_anchor = np.dot(sky_pcs_kl, cube_anchor_l)
                sky_pcs_anchor_cube = sky_pc_anchor.reshape(cube_anchor.shape[0],
                                                        cube_anchor.shape[1],
                                                        cube_anchor.shape[2])
            
                sky_pcs_boat_cube = np.dot(sky_pcs_kl, cube_boat_l).reshape(cube_anchor.shape[0],
                                                                     cube_anchor.shape[1],
                                                                     cube_anchor.shape[2])
                #print(sky_pcs_boat_cube.shape)
                transf_sci = np.zeros((cube_anchor.shape[0]))  #sky number image, science number image
               
                transf_sci = np.inner(sky_pc_anchor, frame_anchor_l[0].T)

                #print(transf_sci.shape)
                #print(transf_sci)
                Msky_pcs_anchor = prepare_matrix(sky_pcs_anchor_cube, scaling=None,
                                             verbose=False)

                mat_inv = np.linalg.inv(np.dot(Msky_pcs_anchor, Msky_pcs_anchor.T))
                transf_sci_scaled = np.dot(mat_inv, transf_sci)
                #print(transf_sci_scaled)
            
                tmp_sky = np.zeros_like(cube[0])
                for n in range(np.max(ncomp)):
                    tmp_sky += np.array([transf_sci_scaled[n]*sky_pcs_boat_cube[n]]).reshape(cube.shape[1], cube.shape[2])
                    position = center - angle_list[k]
                    boat_final = mask_local_boat(mask_annulus, position, angular_width)
                    if n+1 in ncomp:
                        index = np.where(ncomp == n+1)[0][0]
                        sci_cube_skysub[index,k] = frame_boat - tmp_sky
                        
                result_noder[:,k,:,:] += sci_cube_skysub[:,k] * boat_final
                    
        if verbose:
            print("segment {} done".format(i))
                    
        #derotation
        if full_output == False:
            cube_der = np.zeros_like(sci_cube_skysub)
            for n in range(nnpcs):
                cube_der[n,:,:,:] = cube_derotate(
                    sci_cube_skysub[n],
                    angle_list,
                    nproc=nproc,
                    imlib=imlib,
                    interpolation=interpolation,
                    **rot_options,
                )
                results[i,n,:,:] = cube_collapse(cube_der[n,:,:,:], mode=collapse, w=weights)
                results[i,n,:,:] *= boat
        
        
    results = np.sum(results, axis = 0)

    if len(ncomp) == 1:
        results = results[0,:,:]
    #ncomp is a list or not??
    if full_output:
        return result_noder
    else:
        return results


def pca_ardi_mask_dist(
    cube,
    angle_list,
    inner_radius,
    fwhm,
    asize,
    segment_width,
    location,
    n_segments = 8,
    step_location = 1,
    overlap = False,
    wedge = (0, 360),
    mask_rdi = None,
    pa_thr=1,
    ncomp=1,
    svd_mode="lapack",       
    nproc=None,
    min_frames_lib=2,
    max_frames_lib=200,
    tol=1e-1,
    scaling=None,
    imlib="vip-fft",
    interpolation="lanczos4",
    collapse="median",
    full_output=False,
    verbose=1,
    cube_ref=None,
    theta_init=0,
    weights=None,
    cube_sig=None,
    left_eigv=False,
    **rot_options,
):
    """
    segment_width in fwhm
    asize and inner_radius in pixels
    location in degrees
    step_location in degrees
    """
    array = cube
    if array.ndim != 3:
        raise TypeError("Input array is not a cube or 3d array")
    if array.shape[0] != angle_list.shape[0]:
        raise TypeError("Input vector or parallactic angles has wrong length")

    n, y, x = array.shape

    angle_list = check_pa_vector(angle_list)
    if np.isscalar(ncomp):
        ncomp = np.array([ncomp])
    if isinstance(ncomp, list):
        ncomp = np.array(ncomp)
    
    angular_range = wedge[1]-wedge[0] #in degrees
    ann_center = inner_radius+(asize/2)
    perimeter = np.deg2rad(angular_range)*ann_center  #in pixels
    segment_width = fwhm * segment_width
    angular_width = segment_width/ann_center #in radian
    angle_section = np.rad2deg(angular_width)

    if location == 'all':
        if overlap == False:
            if perimeter % segment_width == 0:
                n_images = int(perimeter/segment_width)
                angle = angle_section
            else:
                n_images = int(perimeter/segment_width + 1)
                new_perim = segment_width * n_images
                factor = perimeter/new_perim
                angle = factor*angle_section
        elif overlap == True:
            if segment_width < 1.5*fwhm:
                raise ValueError('Segment width must be greater than 1.5 fwhm ' +
                                 'with overlap = True')
            n_images_no_overlap = int(perimeter/segment_width + 1)
            new_perim = (segment_width+fwhm) * n_images_no_overlap
            n_images = int(new_perim/segment_width + 1)
            factor = perimeter/new_perim
            angle = factor*angle_section

        centers = [wedge[0] + angle/2]
        while (centers[-1]+angle) < wedge[1]:
            centers.append(centers[-1]+angle)
    elif location == 'full':
        angle = angular_range/n_segments
        centers = [wedge[0] + angle/2]
        while (centers[-1]+angle) < wedge[1]:
            centers.append(centers[-1]+angle)
    elif np.isscalar(location):
        centers = [location]
        angle = step_location
        add = 0
        while add+step_location < wedge[1]-wedge[0]:
            centers.append((centers[-1]+step_location)%(wedge[1]-wedge[0]))
            add += step_location

    n_images = len(centers)
    
    mask_annulus = np.ones_like(cube[0], dtype = int)
    mask_annulus = mask_circle(mask_annulus, inner_radius)
    mask_annulus = mask_circle(mask_annulus, inner_radius + asize, mode = 'out')
    cube = cube * mask_annulus
    
    yy, xx = np.where(mask_annulus == 1)
    
    #then convert to bool, boucle on centers, get mask of zone to subtract, rotate it per image...
    cy, cx = frame_center(cube[0])
    twopi = 2*np.pi
    
    masks_centers = []
    
    nnpcs = ncomp.shape[0]
    
    results = np.zeros((n_images, nnpcs, cube.shape[1], cube.shape[2]))
    for i, center in enumerate(centers):
        anchor = mask_exclude(mask_annulus, center, angular_width)
        boat = mask_annulus
        
        sci_cube_skysub = np.zeros((nnpcs, cube.shape[0], cube.shape[1], cube.shape[2]))
        for k in range(cube.shape[0]):
            position = center - angle_list[k]
            anchor_k = mask_annulus - mask_exclude(mask_annulus, position, angle)
            
            if pa_thr != 0:
                indices = np.hstack((_find_indices_adi2(angle_list, k, pa_thr),k))
                indices = np.sort(indices)
            else:
                indices = np.arange(0, cube.shape[0])
            
            ind = np.where(indices == k)[0][0]
            
            cube_used = cube[indices]
            
            cube_anchor = cube_used * anchor_k
            cube_anchor_l = prepare_matrix(cube_anchor, scaling=None, verbose=False)
            cube_boat = prepare_matrix(cube_used * mask_annulus, scaling=None, verbose=False)
            frame_boat = cube[k]
            
            sky_kl = np.dot(cube_anchor_l, cube_anchor_l.T)
            #print(sky_kl)
            #print(sky_kl.shape)
            Msky_kl = prepare_matrix(sky_kl, scaling=None, verbose=False)
            sky_pcs = svd_wrapper(Msky_kl, 'lapack', sky_kl.shape[0], False)
            sky_pcs_kl = sky_pcs.reshape(sky_kl.shape[0], sky_kl.shape[1])
            
            sky_pc_anchor = np.dot(sky_pcs_kl, cube_anchor_l)
            sky_pcs_anchor_cube = sky_pc_anchor.reshape(cube_anchor.shape[0],
                                                        cube_anchor.shape[1],
                                                        cube_anchor.shape[2])
            
            sky_pcs_boat_cube = np.dot(sky_pcs_kl, cube_boat).reshape(cube_anchor.shape[0],
                                                                     cube_anchor.shape[1],
                                                                     cube_anchor.shape[2])
            #print(sky_pcs_boat_cube.shape)
            transf_sci = np.zeros((cube_anchor.shape[0], cube_anchor.shape[0]))
            for j in range(cube_anchor.shape[0]):
                transf_sci[:, j] = np.inner(sky_pc_anchor, cube_anchor_l[j].T)

            #print(transf_sci.shape)
            #print(transf_sci)
            Msky_pcs_anchor = prepare_matrix(sky_pcs_anchor_cube, scaling=None,
                                             verbose=False)

            mat_inv = np.linalg.inv(np.dot(Msky_pcs_anchor, Msky_pcs_anchor.T))
            transf_sci_scaled = np.dot(mat_inv, transf_sci)
            #print(transf_sci_scaled)
            
            tmp_sky = np.zeros_like(cube[k])
            for n in range(np.max(ncomp)):
                tmp_sky += np.array([transf_sci_scaled[n, ind]*sky_pcs_boat_cube[n]]).reshape(cube.shape[1], cube.shape[2])
                if n+1 in ncomp:
                    index = np.where(ncomp == n+1)[0][0]
                    sci_cube_skysub[index,k] = cube[k] - tmp_sky
                    
        #derotation
        cube_der = np.zeros_like(sci_cube_skysub)
        for n in range(nnpcs):
            cube_der[n,:,:,:] = cube_derotate(
                sci_cube_skysub[n],
                angle_list,
                nproc=nproc,
                imlib=imlib,
                interpolation=interpolation,
                **rot_options,
            )
            results[i, n, :, :] = cube_collapse(cube_der[n,:,:,:], mode=collapse, w=weights)
        
        
            
        #accelerate oit if delta_rot = 0, calculate only once principal components
    #n_segments???
    #ncomp is a list or not??
    return results, centers



def pca_ardi_mask_location(
    cube,
    angle_list,
    inner_radius,
    fwhm,
    asize,
    segment_width,
    location,
    overlap = False,
    wedge = (0, 360),
    mask_rdi = None,
    pa_thr=1,
    ncomp=1,
    svd_mode="lapack",       
    nproc=None,
    min_frames_lib=2,
    max_frames_lib=200,
    tol=1e-1,
    scaling=None,
    imlib="vip-fft",
    interpolation="lanczos4",
    collapse="median",
    full_output=False,
    verbose=1,
    cube_ref=None,
    theta_init=0,
    weights=None,
    cube_sig=None,
    left_eigv=False,
    **rot_options,
):
    """
    segment_width in fwhm
    asize and inner_radius in pixels
    location in degrees
    """
    array = cube
    if array.ndim != 3:
        raise TypeError("Input array is not a cube or 3d array")
    if array.shape[0] != angle_list.shape[0]:
        raise TypeError("Input vector or parallactic angles has wrong length")

    n, y, x = array.shape

    angle_list = check_pa_vector(angle_list)
    if np.isscalar(ncomp):
        ncomp = np.array([ncomp])
    if isinstance(ncomp, list):
        ncomp = np.array(ncomp)
    
    angular_range = wedge[1]-wedge[0] #in degrees
    ann_center = inner_radius+(asize/2)
    perimeter = np.deg2rad(angular_range)*ann_center  #in pixels
    segment_width = fwhm * segment_width
    angular_width = segment_width/ann_center #in radian
    angle_sec = np.rad2deg(angular_width)

    
    if np.isscalar(location):
        n_images = int(1)
        centers = [location]
        angle = angle_sec
    elif isinstance(location, list) or isinstance(location, np.ndarray):
        centers = location
        angle = angle_sec
        n_images = int(np.array(location).shape[0])
            
    mask_annulus = np.ones_like(cube[0], dtype = int)
    mask_annulus = mask_circle(mask_annulus, inner_radius)
    mask_annulus = mask_circle(mask_annulus, inner_radius + asize, mode = 'out')
    cube = cube * mask_annulus
    
    yy, xx = np.where(mask_annulus == 1)
    
    #then convert to bool, boucle on centers, get mask of zone to subtract, rotate it per image...
    cy, cx = frame_center(cube[0])
    twopi = 2*np.pi
    
    masks_centers = []
    
    nnpcs = ncomp.shape[0]
    
    results = np.zeros((n_images, nnpcs, cube.shape[1], cube.shape[2]))
    for i, center in enumerate(centers):
        anchor = mask_exclude(mask_annulus, center, angle)
        boat = mask_annulus
        
        sci_cube_skysub = np.zeros((nnpcs, cube.shape[0], cube.shape[1], cube.shape[2]))
        for k in range(cube.shape[0]):
            position = center - angle_list[k]
            anchor_k = mask_annulus - mask_exclude(mask_annulus, position, angle)
            
            if pa_thr != 0:
                indices = np.hstack((_find_indices_adi2(angle_list, k, pa_thr),k))
                indices = np.sort(indices)
            else:
                indices = np.arange(0, cube.shape[0])
            
            ind = np.where(indices == k)[0][0]
            
            cube_used = cube[indices]
            
            cube_anchor = cube_used * anchor_k
            cube_anchor_l = prepare_matrix(cube_anchor, scaling=None, verbose=False)
            cube_boat = prepare_matrix(cube_used * mask_annulus, scaling=None, verbose=False)
            frame_boat = cube[k]
            
            sky_kl = np.dot(cube_anchor_l, cube_anchor_l.T)
            #print(sky_kl)
            #print(sky_kl.shape)
            Msky_kl = prepare_matrix(sky_kl, scaling=None, verbose=False)
            sky_pcs = svd_wrapper(Msky_kl, 'lapack', sky_kl.shape[0], False)
            sky_pcs_kl = sky_pcs.reshape(sky_kl.shape[0], sky_kl.shape[1])
            
            sky_pc_anchor = np.dot(sky_pcs_kl, cube_anchor_l)
            sky_pcs_anchor_cube = sky_pc_anchor.reshape(cube_anchor.shape[0],
                                                        cube_anchor.shape[1],
                                                        cube_anchor.shape[2])
            
            sky_pcs_boat_cube = np.dot(sky_pcs_kl, cube_boat).reshape(cube_anchor.shape[0],
                                                                     cube_anchor.shape[1],
                                                                     cube_anchor.shape[2])
            #print(sky_pcs_boat_cube.shape)
            transf_sci = np.zeros((cube_anchor.shape[0], cube_anchor.shape[0]))
            for j in range(cube_anchor.shape[0]):
                transf_sci[:, j] = np.inner(sky_pc_anchor, cube_anchor_l[j].T)

            #print(transf_sci.shape)
            #print(transf_sci)
            Msky_pcs_anchor = prepare_matrix(sky_pcs_anchor_cube, scaling=None,
                                             verbose=False)

            mat_inv = np.linalg.inv(np.dot(Msky_pcs_anchor, Msky_pcs_anchor.T))
            transf_sci_scaled = np.dot(mat_inv, transf_sci)
            #print(transf_sci_scaled)
            
            tmp_sky = np.zeros_like(cube[k])
            for n in range(np.max(ncomp)):
                tmp_sky += np.array([transf_sci_scaled[n, ind]*sky_pcs_boat_cube[n]]).reshape(cube.shape[1], cube.shape[2])
                if n+1 in ncomp:
                    index = np.where(ncomp == n+1)[0][0]
                    sci_cube_skysub[index,k] = cube[k] - tmp_sky
                    
        #derotation
        cube_der = np.zeros_like(sci_cube_skysub)
        for n in range(nnpcs):
            cube_der[n,:,:,:] = cube_derotate(
                sci_cube_skysub[n],
                angle_list,
                nproc=nproc,
                imlib=imlib,
                interpolation=interpolation,
                **rot_options,
            )
            results[i, n, :, :] = cube_collapse(cube_der[n,:,:,:], mode=collapse, w=weights)
        
            
        #accelerate oit if delta_rot = 0, calculate only once principal components
    #n_segments???
    #ncomp is a list or not??
    return results




def pca_circle_mask(
    cube,
    angle_list,
    fwhm,
    rad,
    theta,
    n_annuli = 'auto',
    mask_radius = 0.75,
    mask_rdi = None,
    asize = 6,
    delta_rot=0,
    ncomp=1,
    svd_mode="lapack",       
    nproc=None,
    min_frames_lib=2,
    max_frames_lib=200,
    tol=1e-1,
    scaling=None,
    imlib="vip-fft",
    interpolation="lanczos4",
    collapse="median",
    full_output=False,
    verbose=1,
    cube_ref=None,
    theta_init=0,
    weights=None,
    cube_sig=None,
    left_eigv=False,
    **rot_options
 ):
    array = cube
    if array.ndim != 3:
        raise TypeError("Input array is not a cube or 3d array")
    if array.shape[0] != angle_list.shape[0]:
        raise TypeError("Input vector or parallactic angles has wrong length")
        
        
    if verbose:
        print("PCA per annulus (or annular sectors):")

    n_im, y, x = array.shape

    angle_list = check_pa_vector(angle_list)
    if np.isscalar(ncomp):
        ncomp = np.array([ncomp])
    if isinstance(ncomp, list):
        ncomp = np.array(ncomp)
    
    
    #then convert to bool, boucle on centers, get mask of zone to subtract, rotate it per image...
    cy, cx = frame_center(cube[0])
    twopi = 2*np.pi
    
    masks_centers = []
    
    nnpcs = ncomp.shape[0]
    if len(ncomp.shape) == 2:
        nnpcs = ncomp.shape[1]
    
    n_images = 1
    results = np.zeros((1, nnpcs, cube.shape[1], cube.shape[2]))
    result_noder = np.zeros((nnpcs,n_im,cube.shape[1],cube.shape[2]))
    
    nbr_frames = []
    
    
    mask_px_rad = mask_radius*fwhm
    
    y_k,x_k = find_coords(rad, (y,x), theta, theta+1, 1)
    aperture = CircularAperture(np.array((x_k, y_k)).T, r=mask_px_rad)
    yy,xx = pxs_coord((y,x), aperture)
    boat = np.zeros_like(cube[0], dtype = int)
    boat[yy[0],xx[0]] = 1
    anchor = np.ones_like(cube[0])
    
    nbr_pixels_c = int(np.sum(boat)-5)
    check_corr = np.zeros((nnpcs, n_im, nbr_pixels_c))
    
    yy,xx = get_annulus_segments(cube[0], rad-mask_px_rad, mask_px_rad*2, 1, 0)[0]
    mask_annulus = np.zeros_like(cube[0], dtype = int)
    mask_annulus[yy,xx] = 1
    
    sci_cube_skysub = np.zeros((nnpcs, cube.shape[0], cube.shape[1], cube.shape[2]))
    for k in range(cube.shape[0]):
        position = theta - angle_list[k]
        y_k,x_k = find_coords(rad, (y,x), position, position+1, 1)
        aperture = CircularAperture(np.array((x_k, y_k)).T, r=mask_px_rad)
        yy,xx = pxs_coord((y,x), aperture)
        boat_k = np.zeros_like(cube[0], dtype = int)
        boat_k[yy[0],xx[0]] = 1
        if mask_rdi is None:
            anchor_k = anchor - boat_k
        elif mask_rdi == 'annulus':
            anchor_k = mask_annulus - boat_k
        
        #boat_k = np.ones_like(cube[0])
        
        pa_thr = _compute_pa_thresh(rad, fwhm, delta_rot)
        
        adi_indices = _find_indices_adi2(angle_list, k, pa_thr,
                                         truncate=True,
                                         max_frames=max_frames_lib)
        adi_indices = np.sort(adi_indices)
        
        frame_boat = cube[k] * boat_k
        frame_anchor = cube[k] * anchor_k
        
        frame_anchor_l = prepare_matrix(frame_anchor[np.newaxis,:,:], scaling=None,
                                            verbose=False)
        
        cube_used = cube[adi_indices]
        if cube_ref is not None:
            cube_used = np.vstack((cube_used, cube_ref))
            
        nbr_frames.append(cube_used.shape[0])
        
        cube_anchor = cube_used * anchor_k
        cube_anchor_l = prepare_matrix(cube_anchor, scaling=None, verbose=False)
    
        cube_boat = cube_used * boat_k
        cube_boat_l = prepare_matrix(cube_boat, scaling=None, verbose=False)
    
        sky_kl = np.dot(cube_anchor_l, cube_anchor_l.T)
        #print(sky_kl)
        #print(sky_kl.shape)
        Msky_kl = prepare_matrix(sky_kl, scaling=None, verbose=False)
        sky_pcs = svd_wrapper(Msky_kl, 'lapack', sky_kl.shape[0], False)
        sky_pcs_kl = sky_pcs.reshape(sky_kl.shape[0], sky_kl.shape[1])
    
        sky_pc_anchor = np.dot(sky_pcs_kl, cube_anchor_l)
        sky_pcs_anchor_cube = sky_pc_anchor.reshape(cube_anchor.shape[0],
                                                cube_anchor.shape[1],
                                                cube_anchor.shape[2])
    
        sky_pcs_boat_cube = np.dot(sky_pcs_kl, cube_boat_l).reshape(cube_anchor.shape[0],
                                                             cube_anchor.shape[1],
                                                             cube_anchor.shape[2])
        #print(sky_pcs_boat_cube.shape)
        transf_sci = np.zeros((cube_anchor.shape[0]))  #sky number image, science number image
       
        transf_sci = np.inner(sky_pc_anchor, frame_anchor_l[0].T)

        #print(transf_sci.shape)
        #print(transf_sci)
        Msky_pcs_anchor = prepare_matrix(sky_pcs_anchor_cube, scaling=None,
                                     verbose=False)

        mat_inv = np.linalg.inv(np.dot(Msky_pcs_anchor, Msky_pcs_anchor.T))
        transf_sci_scaled = np.dot(mat_inv, transf_sci)
        #print(transf_sci_scaled)
    
        tmp_sky = np.zeros_like(cube[0])
        if len(ncomp.shape) == 2:
            for n in range(np.max(ncomp[k])):
                tmp_sky += np.array([transf_sci_scaled[n]*sky_pcs_boat_cube[n]]).reshape(cube.shape[1], cube.shape[2])
                if n+1 in ncomp[k]:
                    index = np.where(ncomp[k] == n+1)[0][0]
                    sci_cube_skysub[index,k] = frame_boat - tmp_sky
        else:
            for n in range(np.max(ncomp)):
                tmp_sky += np.array([transf_sci_scaled[n]*sky_pcs_boat_cube[n]]).reshape(cube.shape[1], cube.shape[2])
                if n+1 in ncomp:
                    index = np.where(ncomp == n+1)[0][0]
                    sci_cube_skysub[index,k] = frame_boat - tmp_sky
                    
                    xx,yy = closest_pixels((y,x), (rad, position), nbr_pixels_c)
                    check_corr[index,k,:] = sci_cube_skysub[index,k,yy,xx]
               
        result_noder[:,k,:,:] += sci_cube_skysub[:,k] * boat_k
        
                
    #derotation
    cube_der = np.zeros_like(sci_cube_skysub)
    check_corr_ = np.zeros((nnpcs, n_im, nbr_pixels_c))
    for n in range(nnpcs):
        cube_der[n,:,:,:] = cube_derotate(
            sci_cube_skysub[n],
            angle_list,
            nproc=nproc,
            imlib=imlib,
            interpolation=interpolation,
            **rot_options,
        )
        results[0,n,:,:] = cube_collapse(cube_der[n,:,:,:], mode=collapse, w=weights)
        results[0,n,:,:] *= boat
        
        xx,yy = closest_pixels((y,x), (rad, theta), nbr_pixels_c)
        check_corr_[n,:,:] = cube_der[n,:,yy,xx].T
    
    
    results = np.sum(results, axis = 0)

    if verbose:
        print(np.mean(nbr_frames))

    if len(ncomp) == 1:
        results = results[0,:,:]
        #ncomp is a list or not??
    if full_output:
        return result_noder, cube_der, results, check_corr, check_corr_
    else:
        return results
    
    
def pca_ardi_annulus_mask_pad(
    cube,
    angle_list,
    inner_radius=4,
    fwhm=4,
    asize=5,
    n_segments = 8,
    segment_side_padding = 0.5,
    segment_radial_padding = 0.2,
    mask_rdi = None,
    pa_thr=0,
    ncomp=1,
    svd_mode="lapack",       
    nproc=None,
    min_frames_lib=2,
    max_frames_lib=200,
    tol=1e-1,
    scaling=None,
    imlib="vip-fft",
    interpolation="lanczos4",
    collapse="median",
    full_output=False,
    verbose=1,
    cube_ref=None,
    theta_init=0,
    weights=None,
    cube_sig=None,
    left_eigv=False,
    **rot_options,
):
    """
    Small boat following the targeted zone
    
    asize and inner_radius in pixels
    location in degrees
    step_location in degrees
    -mask_rdi: location on which project the components. anchor. 
        If None, full-frame
        If mask_rdi == 'annulus', only the rest of the annulus
    -segment_padding: amount of additional space in the mask on all sides of the
    segment being processed. Distance measured in fwhm
    """
    array = cube
    if array.ndim != 3:
        raise TypeError("Input array is not a cube or 3d array")
    if array.shape[0] != angle_list.shape[0]:
        raise TypeError("Input vector or parallactic angles has wrong length")
        
        
    if verbose:
        print("PCA per annulus (or annular sectors):")

    n, y, x = array.shape

    angle_list = check_pa_vector(angle_list)
    if np.isscalar(ncomp):
        ncomp = np.array([ncomp])
    if isinstance(ncomp, list):
        ncomp = np.array(ncomp)
    
    angular_range = 360 #in degrees
    ann_center = inner_radius+(asize/2)

    angular_width = angular_range/n_segments
    centers = [theta_init + angular_width/2]
    total_rot = 0
    while (total_rot+angular_width) < 360:
        total_rot += angular_width
        centers.append(centers[-1]+angular_width)

    n_images = len(centers)
    
    yy,xx = get_annulus_segments(cube[0], inner_radius, asize, 1, 0)[0]
    mask_annulus = np.zeros_like(cube[0], dtype = int)
    mask_annulus[yy,xx] = 1
    
    radial_pad = int(fwhm * segment_radial_padding)
    side_pad = np.rad2deg(np.arctan(segment_side_padding*fwhm/ann_center))
    angular_pad = angular_width + 2*side_pad
    inner_pad = np.max((inner_radius-radial_pad, 0))
    outer_pad = np.min((inner_radius+asize+radial_pad, int(y/2)))
    asize_pad = outer_pad-inner_pad
    yy_b, xx_b = get_annulus_segments(cube[0], inner_pad, asize_pad, 1, 0)[0]
    mask_annulus_pad = np.zeros_like(cube[0], dtype = int)
    mask_annulus_pad[yy_b,xx_b] = 1
    
    yy, xx = np.where(mask_annulus == 1)
    
    #then convert to bool, boucle on centers, get mask of zone to subtract, rotate it per image...
    cy, cx = frame_center(cube[0])
    twopi = 2*np.pi
    
    masks_centers = []
    
    nnpcs = ncomp.shape[0]
    if len(ncomp.shape) == 2:
        nnpcs = ncomp.shape[1]
    
    results = np.zeros((n_images, nnpcs, cube.shape[1], cube.shape[2]))
    result_noder = np.zeros((nnpcs,n,cube.shape[1],cube.shape[2]))
    
    nbr_frames = []
    
    Indices_segments = get_annulus_segments(cube[0], inner_radius, asize, n_segments, theta_init)
    for i, center in enumerate(centers):
        #Project on boat, that is smaller than the area hidden from over-subtraction
        boat = np.zeros_like(cube[0])
        boat[Indices_segments[i][0], Indices_segments[i][1]] = 1
        
        #plot_frames(boat)
        
        sci_cube_skysub = np.zeros((nnpcs, cube.shape[0], cube.shape[1], cube.shape[2]))
        for k in range(cube.shape[0]):
            position = center - angle_list[k]
            boat_k = mask_local_boat(mask_annulus, position, angular_width)
            hidden_anchor_k=mask_local_boat(mask_annulus_pad, position, angular_pad)
            if mask_rdi is None:
                anchor_k = np.ones_like(cube[0]) - hidden_anchor_k
            elif mask_rdi == 'annulus':
                anchor_k = mask_annulus - hidden_anchor_k
            
            #plot_frames(boat_k)
            
            adi_indices = _find_indices_adi2(angle_list, k, pa_thr,
                                             truncate=True,
                                             max_frames=max_frames_lib)
            adi_indices = np.sort(adi_indices)
            
            frame_boat = cube[k] * boat_k
            frame_anchor = cube[k] * anchor_k
            
            frame_anchor_l = prepare_matrix(frame_anchor[np.newaxis,:,:], scaling=None,
                                                verbose=False)
            
            cube_used = cube[adi_indices]
            if cube_ref is not None:
                cube_used = np.vstack((cube_used, cube_ref))
                
            nbr_frames.append(cube_used.shape[0])
            
            cube_anchor = cube_used * anchor_k
            cube_anchor_l = prepare_matrix(cube_anchor, scaling=None, verbose=False)
        
            cube_boat = cube_used * boat_k
            cube_boat_l = prepare_matrix(cube_boat, scaling=None, verbose=False)
        
            sky_kl = np.dot(cube_anchor_l, cube_anchor_l.T)
            #print(sky_kl)
            #print(sky_kl.shape)
            Msky_kl = prepare_matrix(sky_kl, scaling=None, verbose=False)
            sky_pcs = svd_wrapper(Msky_kl, 'lapack', sky_kl.shape[0], False)
            sky_pcs_kl = sky_pcs.reshape(sky_kl.shape[0], sky_kl.shape[1])
        
            sky_pc_anchor = np.dot(sky_pcs_kl, cube_anchor_l)
            sky_pcs_anchor_cube = sky_pc_anchor.reshape(cube_anchor.shape[0],
                                                    cube_anchor.shape[1],
                                                    cube_anchor.shape[2])
        
            sky_pcs_boat_cube = np.dot(sky_pcs_kl, cube_boat_l).reshape(cube_anchor.shape[0],
                                                                 cube_anchor.shape[1],
                                                                 cube_anchor.shape[2])
            #print(sky_pcs_boat_cube.shape)
            transf_sci = np.zeros((cube_anchor.shape[0]))  #sky number image, science number image
           
            transf_sci = np.inner(sky_pc_anchor, frame_anchor_l[0].T)

            #print(transf_sci.shape)
            #print(transf_sci)
            Msky_pcs_anchor = prepare_matrix(sky_pcs_anchor_cube, scaling=None,
                                         verbose=False)

            mat_inv = np.linalg.inv(np.dot(Msky_pcs_anchor, Msky_pcs_anchor.T))
            transf_sci_scaled = np.dot(mat_inv, transf_sci)
            #print(transf_sci_scaled)
        
            tmp_sky = np.zeros_like(cube[0])
            if len(ncomp.shape) == 2:
                for n in range(np.max(ncomp[k])):
                    tmp_sky += np.array([transf_sci_scaled[n]*sky_pcs_boat_cube[n]]).reshape(cube.shape[1], cube.shape[2])
                    if n+1 in ncomp[k]:
                        index = np.where(ncomp[k] == n+1)[0][0]
                        sci_cube_skysub[index,k] = frame_boat - tmp_sky
            else:
                for n in range(np.max(ncomp)):
                    tmp_sky += np.array([transf_sci_scaled[n]*sky_pcs_boat_cube[n]]).reshape(cube.shape[1], cube.shape[2])
                    if n+1 in ncomp:
                        index = np.where(ncomp == n+1)[0][0]
                        sci_cube_skysub[index,k] = frame_boat - tmp_sky
                   
            result_noder[:,k,:,:] += sci_cube_skysub[:,k] * boat_k
            
        if verbose:
            print("segment {} done".format(i))
                    
    #derotation
    if full_output != -1:
        cube_der = np.zeros_like(sci_cube_skysub)
        for n in range(nnpcs):
            cube_der[n,:,:,:] = cube_derotate(
                result_noder[n,:,:,:],
                angle_list,
                nproc=nproc,
                imlib=imlib,
                interpolation=interpolation,
                **rot_options,
            )
            results[i,n,:,:] = cube_collapse(cube_der[n,:,:,:], mode=collapse, w=weights)
            results[i,n,:,:] *= mask_annulus
        
        
    results = np.sum(results, axis = 0)
    
    if verbose:
        print(np.mean(nbr_frames))

    if len(ncomp) == 1:
        results = results[0,:,:]
    #ncomp is a list or not??
    if full_output == -1:
        return result_noder
    elif full_output == True:
        return result_noder, cube_der, results
    else:
        return results
    
    
def pca_segments_mask(
    cube,
    angle_list,
    inner_radius,
    asize,
    fwhm,
    n_segments = 6,
    mask_rdi = None,
    pa_thr=0,
    ncomp=1,
    svd_mode="lapack",       
    nproc=None,
    min_frames_lib=2,
    max_frames_lib=200,
    tol=1e-1,
    scaling=None,
    imlib="vip-fft",
    interpolation="lanczos4",
    collapse="median",
    full_output=False,
    verbose=1,
    cube_ref=None,
    theta_init='auto',
    weights=None,
    cube_sig=None,
    left_eigv=False,
    **rot_options,
):
    """
    Small boat following the targeted zone
    
    asize and inner_radius in pixels
    location in degrees
    step_location in degrees
    -mask_rdi: location on which project the components. anchor. 
        If None, full-frame
        If mask_rdi == 'annulus', only the rest of the annulus
        
    Create two images with offset between locations of boats and anchors
    """
    array = cube
    if array.ndim != 3:
        raise TypeError("Input array is not a cube or 3d array")
    if array.shape[0] != angle_list.shape[0]:
        raise TypeError("Input vector or parallactic angles has wrong length")
        
        
    if verbose:
        print("PCA per annulus (or annular sectors):")

    n, y, x = array.shape

    angle_list = check_pa_vector(angle_list)
    if np.isscalar(ncomp):
        ncomp = np.array([ncomp])
    if isinstance(ncomp, list):
        ncomp = np.array(ncomp)
    
    angular_range = 360 #in degrees
    ann_center = inner_radius+(asize/2)

    angular_width = angular_range/n_segments
    
    if theta_init == 'auto':
        theta_init = [0, angular_width/2]
    
    centers = []
    nbr_offsets = len(theta_init)
    for off in range(nbr_offsets):
        centers.append([theta_init[off] + angular_width/2])
        
    total_rot = 0
    while (total_rot+angular_width) < 360:
        total_rot += angular_width
        for off in range(nbr_offsets):
            centers[off].append(centers[off][-1]+angular_width)

    n_images = len(centers[0])
    
    yy,xx = get_annulus_segments(cube[0], inner_radius, asize, 1, 0)[0]
    mask_annulus = np.zeros_like(cube[0], dtype = int)
    mask_annulus[yy,xx] = 1
    
    #then convert to bool, boucle on centers, get mask of zone to subtract, rotate it per image...
    cy, cx = frame_center(cube[0])
    twopi = 2*np.pi
    
    nnpcs = ncomp.shape[0]
    if len(ncomp.shape) == 2:
        nnpcs = ncomp.shape[1]
    
    results = np.zeros((nbr_offsets,nnpcs,cube.shape[1],cube.shape[2]))
    result_noder = np.zeros((nbr_offsets,nnpcs,n,cube.shape[1],cube.shape[2]))
    cube_der = np.zeros((nbr_offsets,nnpcs,n,cube.shape[1],cube.shape[2]))
    
    nbr_frames = []
    
    for o, off in enumerate(theta_init):
        Indices_segments = get_annulus_segments(cube[0], inner_radius, asize, n_segments, off)
        for i, center in enumerate(centers[o]):
            boat = np.zeros_like(cube[0])
            boat[Indices_segments[i][0], Indices_segments[i][1]] = 1
        
            #plot_frames(boat)
        
            sci_cube_skysub = np.zeros((nnpcs, cube.shape[0], cube.shape[1], cube.shape[2]))
            for k in range(cube.shape[0]):
                position = center - angle_list[k]
                boat_k = mask_local_boat(mask_annulus, position, angular_width)
                if mask_rdi is None:
                    anchor_k = np.ones_like(cube[0]) - boat_k
                elif mask_rdi == 'annulus':
                    anchor_k = mask_annulus - boat_k
            
                #plot_frames(boat_k)
            
                adi_indices = _find_indices_adi2(angle_list, k, pa_thr,
                                             truncate=True,
                                             max_frames=max_frames_lib)
                adi_indices = np.sort(adi_indices)
            
                frame_boat = cube[k] * boat_k
                frame_anchor = cube[k] * anchor_k
            
                frame_anchor_l = prepare_matrix(frame_anchor[np.newaxis,:,:], scaling=None,
                                                verbose=False)
            
                cube_used = cube[adi_indices]
                if cube_ref is not None:
                    cube_used = np.vstack((cube_used, cube_ref))
                
                nbr_frames.append(cube_used.shape[0])
            
                cube_anchor = cube_used * anchor_k
                cube_anchor_l = prepare_matrix(cube_anchor, scaling=None, verbose=False)
        
                cube_boat = cube_used * boat_k
                cube_boat_l = prepare_matrix(cube_boat, scaling=None, verbose=False)
        
                sky_kl = np.dot(cube_anchor_l, cube_anchor_l.T)
                #print(sky_kl)
                #print(sky_kl.shape)
                Msky_kl = prepare_matrix(sky_kl, scaling=None, verbose=False)
                sky_pcs = svd_wrapper(Msky_kl, 'lapack', sky_kl.shape[0], False)
                sky_pcs_kl = sky_pcs.reshape(sky_kl.shape[0], sky_kl.shape[1])
        
                sky_pc_anchor = np.dot(sky_pcs_kl, cube_anchor_l)
                sky_pcs_anchor_cube = sky_pc_anchor.reshape(cube_anchor.shape[0],
                                                    cube_anchor.shape[1],
                                                    cube_anchor.shape[2])
        
                sky_pcs_boat_cube = np.dot(sky_pcs_kl, cube_boat_l).reshape(cube_anchor.shape[0],
                                                                 cube_anchor.shape[1],
                                                                 cube_anchor.shape[2])
                #print(sky_pcs_boat_cube.shape)
                transf_sci = np.zeros((cube_anchor.shape[0]))  #sky number image, science number image
           
                transf_sci = np.inner(sky_pc_anchor, frame_anchor_l[0].T)

                #print(transf_sci.shape)
                #print(transf_sci)
                Msky_pcs_anchor = prepare_matrix(sky_pcs_anchor_cube, scaling=None,
                                         verbose=False)

                mat_inv = np.linalg.inv(np.dot(Msky_pcs_anchor, Msky_pcs_anchor.T))
                transf_sci_scaled = np.dot(mat_inv, transf_sci)
                #print(transf_sci_scaled)
        
                tmp_sky = np.zeros_like(cube[0])
                if len(ncomp.shape) == 2:
                    for n in range(np.max(ncomp[k])):
                        tmp_sky += np.array([transf_sci_scaled[n]*sky_pcs_boat_cube[n]]).reshape(cube.shape[1], cube.shape[2])
                        if n+1 in ncomp[k]:
                            index = np.where(ncomp[k] == n+1)[0][0]
                            sci_cube_skysub[index,k] = frame_boat - tmp_sky
                else:
                    for n in range(np.max(ncomp)):
                        tmp_sky += np.array([transf_sci_scaled[n]*sky_pcs_boat_cube[n]]).reshape(cube.shape[1], cube.shape[2])
                        if n+1 in ncomp:
                            index = np.where(ncomp == n+1)[0][0]
                            sci_cube_skysub[index,k] = frame_boat - tmp_sky
                   
                result_noder[o,:,k,:,:] += sci_cube_skysub[:,k] * boat_k
            
            if verbose:
                print("segment {} done".format(i))
                    
        #derotation
        if full_output != -1:
            for n in range(nnpcs):
                cube_der[o,n,:,:,:] = cube_derotate(
                    result_noder[o,n],
                    angle_list,
                    nproc=nproc,
                    imlib=imlib,
                    interpolation=interpolation,
                    **rot_options,
                    )
                results[o,n,:,:] = cube_collapse(cube_der[o,n,:,:,:], mode=collapse, w=weights)
                results[o,n]*=mask_annulus
    
    if verbose:
        print(np.mean(nbr_frames))

    if len(ncomp) == 1:
        results = results[:,0,:,:]
    #ncomp is a list or not??
    if full_output == -1:
        return result_noder
    elif full_output == True:
        return result_noder, cube_der, results
    else:
        return results
    
    
def pca_annular_mask_edge(
    cube,
    angle_list,
    radius_int,
    fwhm,
    asize,
    n_annuli = 'auto',
    n_segments = 8,
    mask_rdi = None,
    delta_rot=0,
    ncomp=1,
    segment_side_padding = 0,
    segment_radial_padding = 0,
    svd_mode="lapack",       
    nproc=None,
    min_frames_lib=2,
    max_frames_lib=200,
    tol=1e-1,
    scaling=None,
    imlib="vip-fft",
    interpolation="lanczos4",
    collapse="median",
    full_output=False,
    verbose=1,
    cube_ref=None,
    theta_init='auto',
    weights=None,
    cube_sig=None,
    left_eigv=False,
    crop = True,
    **rot_options
 ):
    array = cube
    if array.ndim != 3:
        raise TypeError("Input array is not a cube or 3d array")
    if array.shape[0] != angle_list.shape[0]:
        raise TypeError("Input vector or parallactic angles has wrong length")

    ni, y, x = array.shape
    
    global start_time
    start_time = time_ini()

    angle_list = check_pa_vector(angle_list)
    n_annuli = int((y / 2 - radius_int) / asize)
    
    if isinstance(delta_rot, tuple):
        delta_rot = np.linspace(delta_rot[0], delta_rot[1], num=n_annuli)
    elif np.isscalar(delta_rot):
        delta_rot = [delta_rot] * n_annuli
        
        
    theta_saved = theta_init
    theta_init = []
    if not isinstance(theta_saved, str):
        if isinstance(theta_saved, numbers.Number):
            theta_saved = np.array([theta_saved])
        n_ang = len(theta_saved)
        for ann in range(n_annuli):
            theta_init.append(np.array([theta_saved]))     
    elif theta_saved =='auto':
        n_ang = 2

    if verbose:
        msg = "N annuli = {}, FWHM = {:.3f}"
        print(msg.format(n_annuli, fwhm))
        print("PCA per annulus (or annular sectors):")

    if nproc is None:  # Hyper-threading "duplicates" the cores -> cpu_count/2
        nproc = cpu_count() // 2
        
    if isinstance(ncomp, list):
        nncomp = len(ncomp)
        cube_out = np.zeros([n_ang, nncomp, ni, y,x])
    elif isinstance(ncomp, np.ndarray):
        #1st dim of ncomp is all the epochs
        #2nd dim is the ncomp values tested
        nncomp = ncomp.shape[1]
        cube_out = np.zeros([n_ang, nncomp, ni, y, x])
    elif np.isscalar(ncomp):
        nncomp = 1
        cube_out = np.zeros([n_ang, 1, ni, y, x])
        
    if isinstance(n_segments, list):
        if len(n_segments) != n_annuli:
            raise ValueError('If n_segments is a list, its length must be the same as the number of annuli')
        n_segments_saved = n_segments
    elif n_segments == 'half' or n_segments == 'whole':
        n_segments_saved = np.zeros((n_annuli), dtype = int)
    elif np.isscalar(n_segments):
        n_segments_saved = np.array([n_segments] * n_annuli)
        
    
    for ann in range(n_annuli):
        if isinstance(ncomp, tuple):
            if len(ncomp) == n_annuli:
                ncompann = ncomp[ann]
            else:
                msg = "If `ncomp` is a tuple, its length must match the number "
                msg += "of annuli"
                raise TypeError(msg)
        else:
            ncompann = ncomp

        res_ann_par = _define_annuli(
            angle_list,
            ann,
            n_annuli,
            fwhm,
            radius_int,
            asize,
            delta_rot[ann],
            1,
            verbose,
            True,
        )
        pa_thr, inner_radius, ann_center = res_ann_par
        
        if ann == 0:
            fully_inner_rad = inner_radius
        
        if n_segments == 'half':
            n_segments_saved[ann] = int(np.ceil(4*np.pi*ann_center/fwhm))
        elif n_segments == 'whole':
            n_segments_saved[ann] = int(np.floor(2*np.pi*ann_center/fwhm))
            
        if theta_saved == 'auto':
            angular_range = 360 #in degrees
            angular_width = angular_range/n_segments_saved[ann]
            theta_init.append([0, angular_width/2])
        
        cube_out += pca_ardi_annulus_mask_edge(
            cube,
            angle_list,
            inner_radius,
            fwhm,
            asize,
            n_segments_saved[ann],
            mask_rdi,
            pa_thr,
            ncompann,
            segment_side_padding,
            segment_radial_padding,
            svd_mode,
            nproc,
            min_frames_lib,
            max_frames_lib,
            tol,
            scaling,
            imlib,
            interpolation,
            collapse,
            -1,
            verbose,
            cube_ref,
            theta_init[ann],
            weights,
            cube_sig,
            left_eigv,
            True,
            **rot_options
        )
        
    results = np.zeros((n_ang,nncomp,y,x))
    cube_der = np.zeros_like(cube_out)
    
    #derotation
    for o in range(n_ang):
        for n in range(nncomp):
            cube_der[o, n,:,:,:] = cube_derotate(
                cube_out[o,n],
                angle_list,
                nproc=nproc,
                imlib=imlib,
                interpolation=interpolation,
                **rot_options,
            )
            results[o,n,:,:] = cube_collapse(cube_der[o,n,:,:,:], mode=collapse, w=weights)
        
    final_result = np.zeros((nncomp,y,x))
    final_res_ = np.zeros((nncomp, ni, y, x))
    
    angular_range = 360 #in degrees
    angular_width = angular_range/n_segments_saved

    weights = filled_angular_array(cube[0], n_segments_saved, 
                                   fully_inner_rad, asize, theta_init)
    if len(theta_init[0]) > 1:
        for n in range(nncomp):
            final_result[n]=np.average(results[:,n], axis=0, weights = weights)
            for j in range(ni):
                final_res_[n,j] = np.average(cube_der[:,n,j], axis = 0, weights = weights)
    else:
        final_result[:,:,:] = results[0]
        final_res_ = cube_der[0]
    
        
    if verbose:
        print("Done derotating and combining.")
        timing(start_time)
        
    if nncomp == 1:
        final_result = final_result[0]
        final_res_ = final_res_[0]
        
    if full_output:
        return cube_out, final_res_, final_result
    else:
        return final_result
    

def pca_ardi_annulus_mask_edge(
    cube,
    angle_list,
    inner_radius=4,
    fwhm=4,
    asize=5,
    n_segments = 8,
    mask_rdi = None,
    pa_thr=0,
    ncomp=1,
    segment_side_padding = 0.5,
    segment_radial_padding = 0.2,
    svd_mode="lapack",       
    nproc=None,
    min_frames_lib=2,
    max_frames_lib=200,
    tol=1e-1,
    scaling=None,
    imlib="vip-fft",
    interpolation="lanczos4",
    collapse="median",
    full_output=False,
    verbose=1,
    cube_ref=None,
    theta_init='auto',
    weights=None,
    cube_sig=None,
    left_eigv=False,
    crop=False,
    **rot_options,
):
    """
    Small boat following the targeted zone
    
    asize and inner_radius in pixels
    location in degrees
    step_location in degrees
    -mask_rdi: location on which project the components. anchor. 
        If None, full-frame
        If mask_rdi == 'annulus', only the rest of the annulus
        
    Create two images with offset between locations of boats and anchors
    """
    array = cube
    if array.ndim != 3:
        raise TypeError("Input array is not a cube or 3d array")
    if array.shape[0] != angle_list.shape[0]:
        raise TypeError("Input vector or parallactic angles has wrong length")
        
        
    if verbose:
        print("PCA per annulus (or annular sectors):")

    ni, y, x = array.shape

    angle_list = check_pa_vector(angle_list)
    if np.isscalar(ncomp):
        ncomp = np.array([ncomp])
    if isinstance(ncomp, list):
        ncomp = np.array(ncomp)
    
    angular_range = 360 #in degrees
    ann_center = inner_radius+(asize/2)
    
    if n_segments == 'half':
        n_segments = int(np.ceil(4*np.pi*ann_center/fwhm))
    elif n_segments == 'whole':
        n_segments = int(np.floor(2*np.pi*ann_center/fwhm))
        
    angular_width = angular_range/n_segments
    
    if theta_init == 'auto':
        theta_init = [0, angular_width/2]
    elif np.isscalar(theta_init):
        theta_init = np.array([theta_init])
    
    centers = []
    nbr_offsets = len(theta_init)
    for off in range(nbr_offsets):
        centers.append([theta_init[off] + angular_width/2])
        
    total_rot = 0
    while (total_rot+angular_width) < 359.999999:
        total_rot += angular_width
        for off in range(nbr_offsets):
            centers[off].append(centers[off][-1]+angular_width)

    n_images = len(centers[0])
    
    nnpcs = ncomp.shape[0]
    if len(ncomp.shape) == 2:
        nnpcs = ncomp.shape[1]
    
    results = np.zeros((nbr_offsets,nnpcs,cube.shape[1],cube.shape[2]))
    cube_der = np.zeros((nbr_offsets,nnpcs,ni,cube.shape[1],cube.shape[2]))
    
    yy,xx = get_annulus_segments(cube[0], inner_radius, asize, 1, 0)[0]
    full_mask_annulus = np.zeros_like(cube[0], dtype = int)
    full_mask_annulus[yy,xx] = 1
    
    if crop:
        new_size = int(np.ceil(inner_radius + asize + segment_radial_padding*fwhm)*2)
        if (new_size%2==0) != (y%2==0):
            new_size += 1
        if new_size > y:
            new_size = y
        elif new_size < y:
            cube = cube_crop_frames(cube, new_size, verbose = False)
        in_start = int((y - new_size)/2)
        in_end = int(y-in_start)
    else:
        new_size = y
        in_start = int(0)
        in_end = int(y)
        
    result_noder = np.zeros((nbr_offsets,nnpcs,ni,cube.shape[1],cube.shape[2]))
    
    yy,xx = get_annulus_segments(cube[0], inner_radius, asize, 1, 0)[0]
    mask_annulus = np.zeros_like(cube[0], dtype = int)
    mask_annulus[yy,xx] = 1
    
    radial_pad = int(fwhm * segment_radial_padding)
    side_pad = np.rad2deg(np.arctan(segment_side_padding*fwhm/ann_center))
    angular_pad = angular_width + 2*side_pad
    inner_pad = np.max((inner_radius-radial_pad, 0))
    outer_pad = np.min((inner_radius+asize+radial_pad, int(y/2)))
    asize_pad = outer_pad-inner_pad
    yy_b, xx_b = get_annulus_segments(cube[0], inner_pad, asize_pad, 1, 0)[0]
    mask_annulus_pad = np.zeros_like(cube[0], dtype = int)
    mask_annulus_pad[yy_b,xx_b] = 1
    
    yy, xx = np.where(mask_annulus == 1)
    
    #then convert to bool, boucle on centers, get mask of zone to subtract, rotate it per image...
    cy, cx = frame_center(cube[0])
    twopi = 2*np.pi
    
    nbr_frames = []
    
    for o, off in enumerate(theta_init):
        Indices_segments = get_annulus_segments(cube[0], inner_radius, asize, int(n_segments), off)
        for i, center in enumerate(centers[o]):
            boat = np.zeros_like(cube[0])
            boat[Indices_segments[i][0], Indices_segments[i][1]] = 1
        
            #plot_frames(boat)
        
            sci_cube_skysub = np.zeros((nnpcs, ni, new_size, new_size))
            for k in range(cube.shape[0]):
                position = center - angle_list[k]
                boat_k = mask_local_boat(mask_annulus, position, angular_width)
                hidden_anchor_k=mask_local_boat(mask_annulus_pad, position, angular_pad)
                if mask_rdi is None:
                    anchor_k = np.ones_like(cube[0]) - hidden_anchor_k
                elif mask_rdi == 'annulus':
                    anchor_k = mask_annulus - hidden_anchor_k
            
                #plot_frames(boat_k)
            
                adi_indices = _find_indices_adi2(angle_list, k, pa_thr,
                                             truncate=True,
                                             max_frames=max_frames_lib)
                adi_indices = np.sort(adi_indices)
            
                frame_boat = cube[k] * boat_k
                frame_anchor = cube[k] * anchor_k
            
                frame_anchor_l = prepare_matrix(frame_anchor[np.newaxis,:,:], scaling=None,
                                                verbose=False)
            
                cube_used = cube[adi_indices]
                if cube_ref is not None:
                    cube_used = np.vstack((cube_used, cube_ref))
                
                nbr_frames.append(cube_used.shape[0])
            
                cube_anchor = cube_used * anchor_k
                cube_anchor_l = prepare_matrix(cube_anchor, scaling=None, verbose=False)
        
                cube_boat = cube_used * boat_k
                cube_boat_l = prepare_matrix(cube_boat, scaling=None, verbose=False)
        
                sky_kl = np.dot(cube_anchor_l, cube_anchor_l.T)
                #print(sky_kl)
                #print(sky_kl.shape)
                Msky_kl = prepare_matrix(sky_kl, scaling=None, verbose=False)
                sky_pcs = svd_wrapper(Msky_kl, 'lapack', sky_kl.shape[0], False)
                sky_pcs_kl = sky_pcs.reshape(sky_kl.shape[0], sky_kl.shape[1])
        
                sky_pc_anchor = np.dot(sky_pcs_kl, cube_anchor_l)
                sky_pcs_anchor_cube = sky_pc_anchor.reshape(cube_anchor.shape[0],
                                                    cube_anchor.shape[1],
                                                    cube_anchor.shape[2])
        
                sky_pcs_boat_cube = np.dot(sky_pcs_kl, cube_boat_l).reshape(cube_anchor.shape[0],
                                                                 cube_anchor.shape[1],
                                                                 cube_anchor.shape[2])
                #print(sky_pcs_boat_cube.shape)
                transf_sci = np.zeros((cube_anchor.shape[0]))  #sky number image, science number image
           
                transf_sci = np.inner(sky_pc_anchor, frame_anchor_l[0].T)

                #print(transf_sci.shape)
                #print(transf_sci)
                Msky_pcs_anchor = prepare_matrix(sky_pcs_anchor_cube, scaling=None,
                                         verbose=False)

                mat_inv = np.linalg.inv(np.dot(Msky_pcs_anchor, Msky_pcs_anchor.T))
                transf_sci_scaled = np.dot(mat_inv, transf_sci)
                #print(transf_sci_scaled)
        
                tmp_sky = np.zeros_like(cube[0])
                if len(ncomp.shape) == 2:
                    for n in range(np.max(ncomp[k])):
                        tmp_sky += np.array([transf_sci_scaled[n]*sky_pcs_boat_cube[n]]).reshape(new_size, new_size)
                        if n+1 in ncomp[k]:
                            index = np.where(ncomp[k] == n+1)[0][0]
                            sci_cube_skysub[index,k] = frame_boat - tmp_sky
                else:
                    for n in range(np.max(ncomp)):
                        tmp_sky += np.array([transf_sci_scaled[n]*sky_pcs_boat_cube[n]]).reshape(new_size, new_size)
                        if n+1 in ncomp:
                            index = np.where(ncomp == n+1)[0][0]
                            sci_cube_skysub[index,k] = frame_boat - tmp_sky
                   
                result_noder[o,:,k,:,:] += sci_cube_skysub[:,k] * boat_k
            
            if verbose:
                print("segment {} done".format(i))
                    
        #derotation
        if full_output != -1:
            for n in range(nnpcs):
                cube_der[o,n,:,in_start:in_end,in_start:in_end] = cube_derotate(
                    result_noder[o,n],
                    angle_list,
                    nproc=nproc,
                    imlib=imlib,
                    interpolation=interpolation,
                    **rot_options,
                    )
                results[o,n,:,:] = cube_collapse(cube_der[o,n,:,:,:], mode=collapse, w=weights)
                results[o,n]*=full_mask_annulus
    
    if full_output != -1:
        weights = filled_angular_annulus(cube[0], multiple = 360/n_segments, offset = theta_init)
        final_res = np.zeros((nnpcs, y, x))
        final_ = np.zeros((nnpcs,ni,y,x))
        if len(theta_init) > 1:
            for n in range(nnpcs):
                final_res[n]=np.average(results[:,n], axis=0, weights = weights)
                for j in range(ni):
                    final_[n,j] = np.average(cube_der[:,n,j], axis = 0, weights = weights)
        else:
            final_res[:,:,:] = results[0]
            final_ = cube_der[0]
            
    final_noder = np.zeros_like(cube_der)
    final_noder[:,:,:,in_start:in_end,in_start:in_end] = result_noder
    
    if verbose:
        print(np.mean(nbr_frames))

    if len(ncomp) == 1:
        results = results[:,0,:,:]
    #ncomp is a list or not??
    if full_output == -1:
        return final_noder
    elif full_output == True:
        return result_noder, final_, final_res
    else:
        return results, final_res
    
def threshold_function(value, threshold=0.5):
        """
        Returns 0 if the value is smaller than the threshold, and 1 if the value is greater than or equal to the threshold.
        """
        if value < threshold:
            return 0
        else:
            return 1
        
def smoothstep(x, edge0, edge1):
    """
    Performs a smoothstep interpolation between 0 and 1 for the input value x.

    Parameters:
    x (float): The value to evaluate.
    edge0 (float): The lower edge of the threshold range.
    edge1 (float): The upper edge of the threshold range.

    Returns:
    float: The smoothstep value between 0 and 1.
    """
    # Clamp x to the range [edge0, edge1]
    x = np.clip((x - edge0) / (edge1 - edge0), 0.0, 1.0)
    # Apply the smoothstep function
    return x * x * (3 - 2 * x)


def Quadrant_selection(angle, multiple, smoothing, offset):
    """
    Returns 1 or 0 depending on in which section of the image it is
    
    Parameters:
    array (numpy.ndarray): The 2D array to fill.
    center (tuple): The (x, y) coordinates of the center.

    Returns:
    numpy.ndarray: The filled 2D array.
    """
    def proximity_to_multiple(angle, multiple):
        return np.min([np.abs(angle % multiple), np.abs(multiple - angle % multiple)])
    
    max_dist = (multiple/2)
    
    weights = []
    offset = np.array(offset)
    offset+=90 #For angle to start on right axis
    for o in offset:
        ref_a = angle + o
        proximity = proximity_to_multiple(ref_a, multiple)
        
        if smoothing == 0.5:
            w = threshold_function(proximity/max_dist, 0.5)
        else:
            w = smoothstep(proximity/max_dist, edge0 = 1-smoothing, edge1 = smoothing)
        
        weights.append(w)

    return weights



def filled_angular_annulus(array, multiple, offset, smoothing=0.7):
    """
    Fill a 2D array with values based on the position angle of the pixels.
    Pixels close to an angle multiple of 10 degrees get high values, 
    while those farthest from a multiple of 10 degrees get low values.

    Parameters:
    array (numpy.ndarray): The 2D array to fill.
    center (tuple): The (x, y) coordinates of the center.

    Returns:
    numpy.ndarray: The filled 3D array.
    First array contains the weights for closest to multiple+multiple/2
    Second array contains the weights for closest to the multiple
    """
    rows, cols = array.shape
    filled_array = np.zeros((2, rows, cols))
    smoothing = smoothing/2 + 0.5
    
    center = (rows/2, cols/2)

    def calculate_angle(x, y, center):
        dx = x - center[0]
        dy = y - center[1]
        angle = np.arctan2(dy, dx) * (180 / np.pi)
        angle = (angle + 360) % 360  # Normalize angle to range [0, 360)
        return angle
    
    max_dist = (multiple/2)
    for i in range(rows):
        for j in range(cols):
            angle = calculate_angle(i, j, center)
            filled_array[:,i, j] = Quadrant_selection(angle, multiple, smoothing, offset)

    return filled_array


def filled_angular_array(array, n_segments, inner_radius, asize, offset, smoothing=0.7):
    rows, cols = array.shape
    results = np.zeros((2,rows, cols))
    
    n_annuli = len(n_segments)
    n_segments = np.array(n_segments)
    multiple = 360/n_segments
    
    for ann in range(n_annuli):
        yy,xx = get_annulus_segments(array, inner_radius+ann*asize, asize, 1, 0)[0]
        mask_annulus = np.zeros_like(array, dtype = int)
        mask_annulus[yy,xx] = 1
        if ann == 0:
            mask_annulus = mask_circle(mask_annulus, inner_radius, 
                                       fillwith = 1, mode = 'in')
        elif ann == n_annuli-1:
            mask_annulus = mask_circle(mask_annulus, inner_radius+ann*asize, 
                                       fillwith = 1, mode = 'out')
        this_ang = filled_angular_annulus(array, multiple[ann], offset[ann], smoothing)
        results += this_ang*mask_annulus
    
    return results


def recombine_multiple(images, weights, axis):
    """
    images contains 3D or 4D cubes that need to be recombined.
    The first dimension of images is the axis along which it must be recombined

    """
    
    result = np.average(images, axis = 0, weights = weights)
    
    return result


def find_coords(distance, im_shape, init_angle=0, fin_angle=360, npoints=360, sep = None):
    angular_range = fin_angle - init_angle
    if npoints is None:
        npoints = (np.deg2rad(angular_range) * distance) / sep  # (2*np.pi*rad)/sep
    ang_step = angular_range / npoints  # 360/npoints
    x = []
    y = []
    ang = []
    for i in range(int(npoints)):
        ang.append(ang_step * i + init_angle)
        newx = distance * np.cos(np.deg2rad(ang_step * i + init_angle))
        newy = distance * np.sin(np.deg2rad(ang_step * i + init_angle))
        x.append(newx)
        y.append(newy)
        
    centery, centerx = im_shape[0]/2, im_shape[1]/2
    
    y = np.array(y)
    x = np.array(x)
    
    y+=centery
    x+=centerx
    
    return y, x



def pxs_coord(im_shape, apertures):
    
    y_ind = []
    x_ind = []
    
    for ap in apertures:
        aperture_mask = ap.to_mask(method='center')
        mask_data = aperture_mask.to_image(shape=im_shape)
        
        y_indices, x_indices = np.nonzero(mask_data)

        y_indices = np.array(y_indices, dtype = int)
        x_indices = np.array(x_indices, dtype = int)
        
        y_ind.append(y_indices)
        x_ind.append(x_indices)
    
    return y_ind, x_ind


def closest_pixels(shape, center, n):
    """
    Find the coordinates of the n pixels closest to a given center in a 2D array.

    Parameters:
    array (numpy.ndarray): The 2D array of pixels.
    center (tuple): The (x, y) coordinates of the center.
    n (int): The number of closest pixels to return.

    Returns:
    list: A list of tuples representing the coordinates of the n closest pixels.
    """
    # Get the dimensions of the array
    rows, cols = shape[0], shape[1]
    
    # Create a list to hold the distances and their coordinates
    distances = []
    
    cen = (shape[0]/2,shape[1]/2)
    
    radian = np.deg2rad(center[1])
    center = (cen[0]+center[0]*np.cos(radian), cen[1]+center[0]*np.sin(radian))
    
    # Calculate the Euclidean distance for each pixel
    for i in range(rows):
        for j in range(cols):
            distance = np.sqrt((i - center[0])**2 + (j - center[1])**2)
            distances.append((distance, [i, j]))
    
    # Sort the distances
    distances.sort()
    
    # Get the coordinates of the n closest pixels
    closest_coords = np.array([coord for _, coord in distances[:n]])
    
    return closest_coords[:,0], closest_coords[:,1]



def mask_exclude(mask_annulus, center, angle):
    """
    center and angle to be given in degrees

    """
    
    cy, cx = frame_center(mask_annulus)
    twopi = 2*np.pi

    yy, xx = np.where(mask_annulus == 1)
    
    phi_end = np.deg2rad((center +  angle/2)) % twopi
    phi_start = np.deg2rad((center - angle/2)) % twopi
    phi = np.arctan2(yy - cy, xx - cx)
    phirot = phi % twopi
    mask_s = []
    
    if phi_start < 0:
        phi_start += twopi
        mask_s = (phirot >= phi_start) | (phirot <= phi_end)
    elif phi_start > phi_end:
        mask_s = ~((phirot <= phi_start) & (phirot >= phi_end))
    else:
        mask_s = (phirot >= phi_start) & (phirot < phi_end)
        
    mask_zone = np.zeros_like(mask_annulus)
    mask_zone[yy[mask_s], xx[mask_s]] = 1
    
    return np.array(mask_zone, dtype = int)


def mask_local_boat(mask_annulus, center, angle):
    """
    center and angle to be given in degrees
    
    return boat

    """
    
    cy, cx = frame_center(mask_annulus)
    twopi = 2*np.pi

    yy, xx = np.where(mask_annulus == 1)
    
    phi_end = np.deg2rad((center +  angle/2)) % twopi
    phi_start = np.deg2rad((center - angle/2)) % twopi
    phi = np.arctan2(yy - cy, xx - cx)
    phirot = phi % twopi
    mask_s = []
    
    if phi_start < 0:
        phi_start += twopi
        mask_s = (phirot >= phi_start) | (phirot <= phi_end)
    elif phi_start > phi_end:
        mask_s = ~((phirot <= phi_start) & (phirot >= phi_end))
    else:
        mask_s = (phirot >= phi_start) & (phirot < phi_end)
        
    mask_zone = np.zeros_like(mask_annulus)
    mask_zone[yy[mask_s], xx[mask_s]] = 1
    
    return np.array(mask_zone, dtype = int)