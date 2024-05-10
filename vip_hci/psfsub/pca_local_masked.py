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
from .svd import svd_wrapper, SVDecomposer
from .utils_pca import pca_incremental, pca_grid
from ..config import (timing, time_ini, check_enough_memory, Progressbar,
                      check_array)
from ..config.paramenum import (SvdMode, Adimsdi, Interpolation, Imlib, Collapse,
                                ALGO_KEY)
from ..config.utils_conf import pool_map, iterable
from ..config.utils_param import setup_parameters, separate_kwargs_dict
from ..preproc.derotation import _find_indices_adi, _find_indices_adi2, _compute_pa_thresh
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
    
    results = np.zeros((n_images, nnpcs, cube.shape[1], cube.shape[2]))
    
    Indices_segments = get_annulus_segments(cube[0], inner_radius, asize, n_segments, theta_init)
    for i, center in enumerate(centers):
        boat = np.zeros_like(cube[0])
        boat[Indices_segments[i][0], Indices_segments[i][1]] = 1
        
        #plot_frames(boat)
        
        sci_cube_skysub = np.zeros((nnpcs, cube.shape[0], cube.shape[1], cube.shape[2]))
        for k in range(cube.shape[0]):
            position = center - angle_list[k]
            boat_k = mask_local_boat(mask_annulus, position, angular_width+1)
            if mask_rdi is None:
                anchor_k = np.ones_like(cube[0]) - boat_k
            elif mask_rdi == 'annulus':
                anchor_k = mask_annulus - boat_k
            
            boat_k = np.ones_like(cube[0])
            
            adi_indices = _find_indices_adi2(angle_list, k, pa_thr)
            adi_indices = np.sort(adi_indices)
            
            frame_boat = cube[k] * boat_k
            frame_anchor = cube[k] * anchor_k
            
            frame_anchor_l = prepare_matrix(frame_anchor[np.newaxis,:,:], scaling=None,
                                                verbose=False)
            
            cube_used = cube[adi_indices]
            
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
                if n+1 in ncomp:
                    index = np.where(ncomp == n+1)[0][0]
                    sci_cube_skysub[index,k] = frame_boat - tmp_sky
                    
        if verbose:
            print("segment {} done".format(i))
                    
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
            results[i,n,:,:] = cube_collapse(cube_der[n,:,:,:], mode=collapse, w=weights)
            results[i,n,:,:] *= boat
        
        
    results = np.sum(results, axis = 0)
    
    if verbose:
        print("Done derotating and combining.")
        timing(start_time)

    if len(ncomp) == 1:
        results = results[0,:,:]
    #ncomp is a list or not??
    return results



def pca_ardi_annulus_masked(
    cube,
    angle_list,
    inner_radius,
    fwhm,
    asize,
    n_segments = 8,
    mask_rdi = None,
    delta = False,
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
    twopi = 2*np.pi
    
    masks_centers = []
    
    nnpcs = ncomp.shape[0]
    
    results = np.zeros((n_images, nnpcs, cube.shape[1], cube.shape[2]))
    
    Indices_segments = get_annulus_segments(cube[0], inner_radius, asize, n_segments, theta_init)
    for i, center in enumerate(centers):
        boat = np.zeros_like(cube[0])
        boat[Indices_segments[i][0], Indices_segments[i][1]] = 1
        
        plot_frames(boat)
        
        sci_cube_skysub = np.zeros((nnpcs, cube.shape[0], cube.shape[1], cube.shape[2]))
        
        position = center - para_range/2 - angle_list[0]
        boat_k = mask_local_boat(mask_annulus, position, angular_width+para_range)
        if mask_rdi is None:
            anchor_k = np.ones_like(cube[0]) - boat_k
        elif mask_rdi == 'annulus':
            anchor_k = mask_annulus - boat_k
            
        plot_frames(anchor_k)
        
        boat_k = np.ones_like(cube[0])
        
        if pa_thr == 0:
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
                for n in range(np.max(ncomp)):
                    tmp_sky += np.array(transf_sci_scaled[n, k]*sky_pcs_boat_cube[n]).reshape(cube.shape[1], cube.shape[2])
                    if n+1 in ncomp:
                        index = np.where(ncomp == n+1)[0][0]
                        sci_cube_skysub[index,k] = cube_boat[k] - tmp_sky
                        
        else:
            for k in range(cube.shape[0]):
                adi_indices = _find_indices_adi2(angle_list, k, pa_thr)
                adi_indices = np.sort(adi_indices)
                
                frame_boat = cube[k] * boat_k
                frame_anchor = cube[k] * anchor_k
                
                frame_anchor_l = prepare_matrix(frame_anchor[np.newaxis,:,:], scaling=None,
                                                    verbose=False)
                
                cube_used = cube[adi_indices]
                
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
                    if n+1 in ncomp:
                        index = np.where(ncomp == n+1)[0][0]
                        sci_cube_skysub[index,k] = frame_boat - tmp_sky
                    
        if verbose:
            print("segment {} done".format(i))
                    
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
            results[i,n,:,:] = cube_collapse(cube_der[n,:,:,:], mode=collapse, w=weights)
            results[i,n,:,:] *= boat
        
        
    results = np.sum(results, axis = 0)
    
    if verbose:
        print("Done derotating and combining.")
        timing(start_time)

    if len(ncomp) == 1:
        results = results[0,:,:]
    #ncomp is a list or not??
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