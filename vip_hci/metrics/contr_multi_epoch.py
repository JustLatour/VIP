# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 20:00:53 2023

@author: justi
"""

import numpy as np
import pandas as pd
from inspect import getfullargspec
try:
    from photutils.aperture import aperture_photometry, CircularAperture
except:
    from photutils import aperture_photometry, CircularAperture
from scipy.interpolate import InterpolatedUnivariateSpline
from scipy import stats
from scipy.signal import savgol_filter
from skimage.draw import disk
from matplotlib import pyplot as plt
from ..fm import cube_inject_companions, frame_inject_companion, normalize_psf
from ..config import time_ini, timing
from ..config.utils_conf import vip_figsize, vip_figdpi
from ..var import frame_center, dist
from ..preproc.derotation import _define_annuli

from .contrcurve import noise_per_annulus, aperture_flux
from ..psfsub.pca_fullfr import *
from ..psfsub.pca_fullfr import PCA_Params
from ..psfsub.pca_fullfr import PCA_Params
from ..psfsub.pca_local import *
from ..psfsub.pca_local import PCA_ANNULAR_Params
from ..psfsub.pca_multi_epoch import *

from hciplot import plot_frames

def contrast_optimized(
    cube,
    angle_list,
    psf_template,
    fwhm,
    pxscale,
    starphot,
    algo,
    ncomp,
    cube_delimiter,
    cube_ref_delimiter = None,
    sigma=5,
    nbranch=1,
    distance=2,
    theta=0,
    inner_rad=1,
    fc_rad_sep=3,
    noise_sep=1,
    wedge=(0, 360),
    fc_snr=100,
    student=True,
    transmission=None,
    smooth=True,
    interp_order=2,
    plot=True,
    dpi=vip_figdpi,
    debug=False,
    verbose=True,
    full_output=False,
    save_plot=None,
    object_name=None,
    frame_size=None,
    fix_y_lim=(),
    figsize=vip_figsize,
    algo_class=None,
    **algo_dict
):
    
    if cube.ndim != 3 and cube.ndim != 4:
        raise TypeError("The input array is not a 3d or 4d cube")
    if cube.ndim == 3 and (cube.shape[0] != angle_list.shape[0]):
        raise TypeError("Input parallactic angles vector has wrong length")
    if cube.ndim == 4 and (cube.shape[1] != angle_list.shape[0]):
        raise TypeError("Input parallactic angles vector has wrong length")
    if cube.ndim == 3 and psf_template.ndim != 2:
        raise TypeError("Template PSF is not a frame (for ADI case)")
    if cube.ndim == 4 and psf_template.ndim != 3:
        raise TypeError("Template PSF is not a cube (for ADI+IFS case)")
    if transmission is not None:
        if len(transmission) != 2 and len(transmission) != cube.shape[0] + 1:
            msg = "Wrong shape for transmission should be 2xn_rad or (nch+1) "
            msg += "x n_rad, instead of {}".format(transmission.shape)
            raise TypeError(msg)

    if isinstance(fwhm, (np.ndarray, list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if verbose:
        start_time = time_ini()
        if isinstance(starphot, float) or isinstance(starphot, int):
            msg0 = "ALGO : {}, FWHM = {}, # BRANCHES = {}, SIGMA = {},"
            msg0 += " STARPHOT = {}"
            print(msg0.format('pca_annular_multi_epoch', fwhm_med, nbranch, sigma, starphot))
        else:
            msg0 = "ALGO : {}, FWHM = {}, # BRANCHES = {}, SIGMA = {}"
            print(msg0.format('pca_annular_multi_epoch', fwhm_med, nbranch, sigma))

    # throughput
    verbose_thru = False
    if verbose == 2:
        verbose_thru = True
        
        
        
    #Do everything needed on empty cube, only once!
    array = cube
    parangles = angle_list
    nproc = algo_dict.get("nproc", 1)
    imlib = algo_dict.get("imlib", "vip-fft")
    interpolation = algo_dict.get("interpolation", "lanczos4")
    scaling = algo_dict.get("scaling", None)

    if array.ndim != 3 and array.ndim != 4:
        raise TypeError("The input array is not a 3d or 4d cube")

    else:
        if psf_template.shape[1] % 2 == 0:
            raise ValueError("Only odd-sized PSF is accepted")
        if not hasattr(algo, "__call__"):
            raise TypeError("Parameter `algo` must be a callable function")
        if not isinstance(inner_rad, int):
            raise TypeError("inner_rad must be an integer")
        angular_range = wedge[1] - wedge[0]
        if nbranch > 1 and angular_range < 360:
            msg = "Only a single branch is allowed when working on a wedge"
            raise RuntimeError(msg)

    if isinstance(fwhm, (np.ndarray, list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if verbose:
        start_time = time_ini()
    # ***************************************************************************
    # Compute noise in concentric annuli on the "empty frame"

    # TODO: Clean below?
    # Consider 3 cases depending on whether algo is (i) defined externally,
    # (ii) a VIP postproc algorithm; (iii) ineligible for contrast curves
    
    
    NEpochs = len(ncomp)
    SizeEpoch = int(cube.shape[0]/NEpochs)
    SizeEpochs = np.ones(NEpochs, dtype = int)
    CompPerE = np.ones(NEpochs, dtype = int)
    FullSize = int(0)
    NCombinations = int(1)
    SizeImage = int(cube[0].shape[0])
    for i in range(0, NEpochs, 1):
        CompPerE[i] = int(len(ncomp[i]))
        FullSize += int(CompPerE[i])
        NCombinations *= int(CompPerE[i])
        SizeEpochs[i] = cube_delimiter[i+1]-cube_delimiter[i]
    DefaultSizeE = np.max(SizeEpochs)
    Res_fc = np.zeros((FullSize, nbranch, DefaultSizeE, SizeImage, SizeImage), dtype = float)
    Res_no_fc = np.zeros((FullSize, DefaultSizeE, SizeImage, SizeImage), dtype = float)
    frames_no_fc = np.zeros((NCombinations, SizeImage, SizeImage))
    noiseF = np.zeros((NCombinations))
    res_levelF = np.zeros((NCombinations))
    fc_maps = np.zeros((NCombinations, SizeImage, SizeImage))
    
    D = int(distance)
    rad_dist = distance * fwhm
    Throughput = np.zeros((NCombinations, nbranch))
    Thru_Cont = np.zeros((NCombinations, 2))
    
    
    class_params, rot_options = separate_kwargs_dict(
        initial_kwargs=algo_dict, parent_class=PCA_ANNULAR_MULTI_EPOCH_Params)
    

    class_params['cube'] = cube
    class_params['angle_list'] = angle_list
    class_params['fwhm'] = fwhm
    class_params['cube_delimiter'] = cube_delimiter
    class_params['cube_ref_delimiter'] = cube_ref_delimiter
    
    all_args = ()
    algo_params = PCA_ANNULAR_MULTI_EPOCH_Params(*all_args, **class_params)
    
    Inherited, NotInherited = Inherited_Params(algo_params)
    
    ToRemove = ['full_output', 'ncomp', 'cube', 'cube_ref', 'angle_list', 'delta_rot']
    Args_left = RemoveKeys(Inherited, ToRemove)
    
    if (type(algo_params.delta_rot) == float):
        algo_params.delta_rot = np.full_like(np.array(CompPerE), algo_params.delta_rot, dtype = float)
    elif (type(algo_params.delta_rot) == int):
        algo_params.delta_rot = np.full_like(np.array(CompPerE), algo_params.delta_rot, dtype = float)
    elif algo_params.delta_rot == None:
        algo_params.delta_rot = np.full(np.array(CompPerE).shape, algo_params.delta_rot, dtype = None)
    
    
    ReusedRef = False
    if algo_params.cube_ref is not None:    
        #To know the format used for cube_ref_delimiter
        if len(algo_params.cube_ref_delimiter) == 2*NEpochs:
            ReusedRef = True
        
    # We crop the PSF and check if PSF has been normalized (so that flux in
    # 1*FWHM aperture = 1) and fix if needed
    new_psf_size = int(round(3 * fwhm_med))
    if new_psf_size % 2 == 0:
        new_psf_size += 1

    if cube.ndim == 3:
        #Calculation of no_fc comparison
        Index = 0
        if algo_params.cube_ref is not None:                 

            for i in range(0, NEpochs, 1):
                StartIndexCube = algo_params.cube_delimiter[i]
                EndIndexCube = algo_params.cube_delimiter[i+1]
                
                if ReusedRef:
                    StartIndexCubeRef = algo_params.cube_ref_delimiter[2*i]
                    EndIndexCubeRef = algo_params.cube_ref_delimiter[2*i +1]
                
                else:
                    StartIndexCubeRef = algo_params.cube_ref_delimiter[i]
                    EndIndexCubeRef = algo_params.cube_ref_delimiter[i+1]
                
                
                for j in range(0, CompPerE[i], 1):
                    _, residuals_cube_, _ = pca_annular(
                        algo_params.cube[StartIndexCube:EndIndexCube,:,:],
                        algo_params.angle_list[StartIndexCube:EndIndexCube],
                        cube_ref = algo_params.cube_ref[StartIndexCubeRef:EndIndexCubeRef,:,:],
                        ncomp = int(ncomp[i][j]), full_output = True, 
                        delta_rot = algo_params.delta_rot[i],
                        **Args_left, **rot_options)
                    
                    CorrectDimRes = np.zeros((DefaultSizeE, SizeImage, SizeImage), dtype = float)
                    CorrectDimRes[0:SizeEpochs[i], :, :] = residuals_cube_
                    Res_no_fc[Index, :, :, :] = CorrectDimRes
                    Index += 1
        
        
        else:
            for i in range(0, NEpochs, 1):
                StartIndex = algo_params.cube_delimiter[i]
                EndIndex = algo_params.cube_delimiter[i+1]
                
                for j in range(0, CompPerE[i], 1):
                    _, residuals_cube_, _ = pca_annular(
                        algo_params.cube[StartIndex:EndIndex,:,:],
                        algo_params.angle_list[StartIndex:EndIndex],
                        ncomp = int(ncomp[i][j]), full_output = True, 
                        delta_rot = algo_params.delta_rot[i],
                        **Args_left, **rot_options)
                    
                    CorrectDimRes = np.zeros((DefaultSizeE, SizeImage, SizeImage), dtype = float)
                    CorrectDimRes[0:SizeEpochs[i], :, :] = residuals_cube_
                    Res_no_fc[Index, :, :, :] = CorrectDimRes
                    Index += 1
                    
        
        M = np.ones(NEpochs)
        for i in range(0, NEpochs, 1):
            for j in range(NEpochs-1, -1, -1):
                if j >= i:
                    M[i] *= CompPerE[j]
    
        Indices = np.zeros(NEpochs)

        k = 0
        for i in range(0, NCombinations, 1):

            k = 0
            for j in range(0, NEpochs, 1):
                if (i+1)%M[NEpochs-j-1] == 0:
                    k = j+1
            
            
            GlobalResiduals_no_fc = np.array([[[]]])
            Sum = 0
            for j in range(0, NEpochs, 1):
                if j == 0:
                    GlobalResiduals_no_fc = Res_no_fc[int(Indices[0]), 0:SizeEpochs[j]]
                else:
                    Sum += CompPerE[j-1]
                    GlobalResiduals_no_fc = np.vstack((GlobalResiduals_no_fc, Res_no_fc[int(Sum+Indices[j]), 0:SizeEpochs[j]]))
            
            
            frames_no_fc[i,:,:] = np.median(GlobalResiduals_no_fc, axis = 0)
            
            
            noise, res_level, vector_radd = noise_per_annulus(frames_no_fc[i],
                                                              separation=fwhm_med,
                                                              fwhm=fwhm_med,
                                                              wedge=wedge)


            vector_radd = vector_radd[inner_rad - 1:]
            noise = noise[inner_rad - 1:]
            res_level = res_level[inner_rad - 1:]
            if verbose:
                print("Measured annulus-wise noise in resulting frame")
                timing(start_time)
            
            
            noiseF[i] = noise[D-1]+((noise[D]-noise[D-1])/(vector_radd[D]-vector_radd[D-1]))*(distance*fwhm-vector_radd[D-1])
            res_levelF[i] = res_level[D-1]+((res_level[D]-res_level[D-1])/(vector_radd[D]-vector_radd[D-1]))*(distance*fwhm-vector_radd[D-1])
            
            
            Indices[NEpochs-k-1] += 1
            for j in range(0, k, 1):
                Indices[NEpochs-j-1] = 0
    
        
        n, y, x = array.shape
        psf_template = normalize_psf(
            psf_template,
            fwhm=fwhm,
            verbose=verbose,
            size=min(new_psf_size, psf_template.shape[1]),
        )

        # Initialize the fake companions
        angle_branch = angular_range / nbranch
        thruput_arr = np.zeros((nbranch, noise.shape[0]))
        fc_map_all = np.zeros((nbranch * fc_rad_sep, y, x))
        frame_fc_all = np.zeros((nbranch * fc_rad_sep, y, x))
        cy, cx = frame_center(array[0])

        # each branch is computed separately
        for br in range(nbranch):
            # each pattern is computed separately. For each one the companions
            # are separated by "fc_rad_sep * fwhm", interleaving the injections
            cube_fc = array.copy()
            # filling map with small numbers
            fc_map = np.ones_like(array[0]) * 1e-6
            fcy = 0
            fcx = 0
            flux = fc_snr * np.nanmean(noiseF)
            cube_fc = cube_inject_companions(
                cube_fc,
                psf_template,
                parangles,
                flux,
                rad_dists=rad_dist,
                theta=br * angle_branch + theta,
                nproc=nproc,
                imlib=imlib,
                interpolation=interpolation,
                verbose=False,
            )
            
            
            y = cy + rad_dist * \
                np.sin(np.deg2rad(br * angle_branch + theta))
            x = cx + rad_dist * \
                np.cos(np.deg2rad(br * angle_branch + theta))
            fc_map = frame_inject_companion(
                fc_map, psf_template, y, x, flux, imlib, interpolation
            )
            fcy = y
            fcx = x

            if verbose:
                msg2 = "Fake companions injected in branch {} "
                print(msg2.format(br + 1))
                timing(start_time)
            
            
            Index = 0
            if algo_params.cube_ref is not None:                 

                for i in range(0, NEpochs, 1):
                    StartIndexCube = algo_params.cube_delimiter[i]
                    EndIndexCube = algo_params.cube_delimiter[i+1]
                    
                    if ReusedRef:
                        StartIndexCubeRef = algo_params.cube_ref_delimiter[2*i]
                        EndIndexCubeRef = algo_params.cube_ref_delimiter[2*i +1]
                    
                    else:
                        StartIndexCubeRef = algo_params.cube_ref_delimiter[i]
                        EndIndexCubeRef = algo_params.cube_ref_delimiter[i+1]
                        
                    for j in range(0, CompPerE[i], 1):
                        _, residuals_cube_, _ = pca_annular(
                            cube_fc[StartIndexCube:EndIndexCube,:,:],
                            algo_params.angle_list[StartIndexCube:EndIndexCube],
                            cube_ref = algo_params.cube_ref[StartIndexCubeRef:EndIndexCubeRef,:,:],
                            ncomp = int(ncomp[i][j]), full_output = True, 
                            delta_rot = algo_params.delta_rot[i],
                            **Args_left, **rot_options)
                        
                        CorrectDimRes = np.zeros((DefaultSizeE, SizeImage, SizeImage), dtype = float)
                        CorrectDimRes[0:SizeEpochs[i], :, :] = residuals_cube_
                        Res_fc[Index, br, :, :, :] = CorrectDimRes
                        Index += 1
            
            
            else:
                for i in range(0, NEpochs, 1):
                    StartIndex = algo_params.cube_delimiter[i]
                    EndIndex = algo_params.cube_delimiter[i+1]
                    
                    for j in range(0, CompPerE[i], 1):
                        _, residuals_cube_, _ = pca_annular(
                            cube_fc[StartIndex:EndIndex,:,:],
                            algo_params.angle_list[StartIndex:EndIndex],
                            ncomp = int(ncomp[i][j]), full_output = True, 
                            delta_rot = algo_params.delta_rot[i],
                            **Args_left, **rot_options)
                        
                        CorrectDimRes = np.zeros((DefaultSizeE, SizeImage, SizeImage), dtype = float)
                        CorrectDimRes[0:SizeEpochs[i], :, :] = residuals_cube_
                        Res_fc[Index, br, :, :, :] = CorrectDimRes
                        Index += 1
                    
             
    
            M = np.ones(NEpochs)
            for i in range(0, NEpochs, 1):
                for j in range(NEpochs-1, -1, -1):
                    if j >= i:
                        M[i] *= CompPerE[j]
        
            Indices = np.zeros(NEpochs)

            k = 0
            for i in range(0, NCombinations, 1):
    
                k = 0
                for j in range(0, NEpochs, 1):
                    if (i+1)%M[NEpochs-j-1] == 0:
                        k = j+1
                
                
                GlobalResiduals_fc = np.array([[[]]])
                Sum = 0
                for j in range(0, NEpochs, 1):
                    if j == 0:
                        GlobalResiduals_fc = Res_fc[int(Indices[0]), br, 0:SizeEpochs[j]]
                    else:
                        Sum += CompPerE[j-1]
                        GlobalResiduals_fc = np.vstack((GlobalResiduals_fc, Res_fc[int(Sum+Indices[j]), br, 0:SizeEpochs[j]]))
                
                frame_fc = np.median(GlobalResiduals_fc, axis = 0)
                
                Indices[NEpochs-k-1] += 1
                for j in range(0, k, 1):
                    Indices[NEpochs-j-1] = 0
    
    
                injected_flux = apertureOne_flux(fc_map, fcy, fcx, fwhm_med)
                recovered_flux = apertureOne_flux(
                    (frame_fc - frames_no_fc[i]), fcy, fcx, fwhm_med)
                
                thruput = recovered_flux / injected_flux
                
                Throughput[i, br] = thruput
        
        
    rad_samp = rad_dist
    noise_samp = noiseF
    res_lev_samp = res_levelF           
                
    res_lev_samp = np.abs(res_lev_samp)

    noise_samp_sm = noise_samp
    res_lev_samp_sm = res_lev_samp
    
    
    for i in range(0, NCombinations, 1):
        Thru_Cont[i,0] = np.nanmean(Throughput[i,:])


    
    if isinstance(starphot, float) or isinstance(starphot, int):
        Thru_Cont[:,1] = (
            (sigma * noise_samp_sm + res_lev_samp_sm) / Thru_Cont[:,0]
        ) / starphot
    else:
        Thru_Cont[:,1] = (
            (sigma * noise_samp_sm + res_lev_samp_sm) / Thru_Cont[:,0]
        ) / np.median(starphot)
    
    return Thru_Cont


def contr_dist(
    cube,
    angle_list,
    psf_template,
    fwhm,
    distance,
    pxscale,
    starphot,
    algo,
    sigma=5,
    nbranch=1,
    theta=0,
    inner_rad=1,
    fc_rad_sep=3,
    noise_sep=1,
    wedge=(0, 360),
    fc_snr=20,
    student=True,
    transmission=None,
    dpi=vip_figdpi,
    debug=False,
    verbose=True,
    full_output=False,
    save_plot=None,
    object_name=None,
    frame_size=None,
    fix_y_lim=(),
    figsize=vip_figsize,
    algo_class=None,
    matrix_adi_ref=None,
    angle_adi_ref=None,
    **algo_dict,
):
    if cube.ndim != 3 and cube.ndim != 4:
        raise TypeError("The input array is not a 3d or 4d cube")
    if cube.ndim == 3 and (cube.shape[0] != angle_list.shape[0]):
        raise TypeError("Input parallactic angles vector has wrong length")
    if cube.ndim == 4 and (cube.shape[1] != angle_list.shape[0]):
        raise TypeError("Input parallactic angles vector has wrong length")
    if cube.ndim == 3 and psf_template.ndim != 2:
        raise TypeError("Template PSF is not a frame (for ADI case)")
    if cube.ndim == 4 and psf_template.ndim != 3:
        raise TypeError("Template PSF is not a cube (for ADI+IFS case)")
    if transmission is not None:
        if len(transmission) != 2 and len(transmission) != cube.shape[0] + 1:
            msg = "Wrong shape for transmission should be 2xn_rad or (nch+1) "
            msg += "x n_rad, instead of {}".format(transmission.shape)
            raise TypeError(msg)

    if isinstance(fwhm, (np.ndarray, list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if verbose:
        start_time = time_ini()
        if isinstance(starphot, float) or isinstance(starphot, int):
            msg0 = "ALGO : {}, FWHM = {}, # BRANCHES = {}, SIGMA = {},"
            msg0 += " STARPHOT = {}"
            print(msg0.format(algo.__name__, fwhm_med, nbranch, sigma, starphot))
        else:
            msg0 = "ALGO : {}, FWHM = {}, # BRANCHES = {}, SIGMA = {}"
            print(msg0.format(algo.__name__, fwhm_med, nbranch, sigma))
            
    if cube.ndim != 3 and cube.ndim != 4:
        raise TypeError("The input array is not a 3d or 4d cube")

    else:
        if psf_template.shape[1] % 2 == 0:
            raise ValueError("Only odd-sized PSF is accepted")
        if not hasattr(algo, "__call__"):
            raise TypeError("Parameter `algo` must be a callable function")
        if not isinstance(inner_rad, int):
            raise TypeError("inner_rad must be an integer")
        angular_range = wedge[1] - wedge[0]
        if nbranch > 1 and angular_range < 360:
            msg = "Only a single branch is allowed when working on a wedge"
            raise RuntimeError(msg)
        
    nproc = algo_dict.get("nproc", 1)
    imlib = algo_dict.get("imlib", "vip-fft")
    interpolation = algo_dict.get("interpolation", "lanczos4")
    scaling = algo_dict.get("scaling", None)
    ncomp = algo_dict.get("ncomp")
    
    if cube.ndim == 3:
        if isinstance(ncomp, list):
            nnpcs = len(ncomp)
        elif isinstance(ncomp, tuple):
            if isinstance(ncomp[1], list):
                nnpcs = len(ncomp[1])
            else:
                nnpcs = 1
        else:
            nnpcs = 1
    elif cube.ndim == 4:
        if isinstance(ncomp, list):
            if len(ncomp) == cube.shape[0]:
                nnpcs = 1
            else:
                nnpcs = len(ncomp)
        else:
            nnpcs = 1
        
    SizeImage = int(cube[0].shape[0])
    frames_fc = np.zeros((nnpcs, nbranch, SizeImage, SizeImage), dtype = float)
    frames_no_fc = np.zeros((nnpcs, SizeImage, SizeImage), dtype = float)
    
    rad_dist = distance * fwhm
    
    if isinstance(fwhm, (np.ndarray, list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if verbose:
        start_time = time_ini()
        
    algo_name = algo.__name__
    idx = algo.__module__.index('.', algo.__module__.index('.') + 1)
    mod = algo.__module__[:idx]
    tmp = __import__(mod, fromlist=[algo_name.upper()+'_Params'])    
    algo_params = getattr(tmp, algo_name.upper()+'_Params')
    
    if matrix_adi_ref is not None:
        if 'cube_ref' in algo_dict.keys() and algo_dict['cube_ref'] is not None:
            NAdiRef = algo_dict['cube_ref'].shape[0]
            algo_dict['cube_ref'] = np.vstack((algo_dict['cube_ref'], matrix_adi_ref))
        else:
            NAdiRef = 0
            algo_dict['cube_ref'] = matrix_adi_ref
        NRefT = algo_dict['cube_ref'].shape[0]
    
    frames_no_fc[:, :, :] = np.array(algo(cube=cube, angle_list=angle_list, fwhm=fwhm_med,
                      verbose=verbose, **algo_dict))

    
    #CHANGE NOISE ANNULI TO HAVE IT ONLY HERE AT THIS DISTANCE !!!
    noise_res = [noise_dist(frames_no_fc[i], rad_dist, fwhm_med, wedge, 
                                  False, debug) for i in range(0, nnpcs)]
    
    noise = np.array(noise_res)[:,0,0]
    mean_res = np.array(noise_res)[:,0,1]
    
    # We crop the PSF and check if PSF has been normalized (so that flux in
    # 1*FWHM aperture = 1) and fix if needed
    new_psf_size = int(round(3 * fwhm_med))
    if new_psf_size % 2 == 0:
        new_psf_size += 1
    
    n, y, x = cube.shape
    psf_template = normalize_psf(
        psf_template,
        fwhm=fwhm,
        verbose=verbose,
        size=min(new_psf_size, psf_template.shape[1]),
    )
    
    
    # Initialize the fake companions
    angle_branch = angular_range / nbranch
    Throughput = np.zeros((nnpcs, nbranch))
    fc_map = np.zeros((nbranch, y, x))
    cy, cx = frame_center(cube[0])
    parangles = angle_list

    # each branch is computed separately
    if matrix_adi_ref is not None:
        copy_ref = np.copy(algo_dict['cube_ref'])
        
    for br in range(nbranch):
        
        if matrix_adi_ref is not None:
            algo_dict['cube_ref'] = np.copy(copy_ref)
        
        # each pattern is computed separately. For each one the companions
        # are separated by "fc_rad_sep * fwhm", interleaving the injections
        fc_map = np.ones_like(cube[0]) * 1e-6
        fcy = 0
        fcx = 0
        flux = fc_snr * np.min(noise)
        
        if matrix_adi_ref is None:
            cube_fc = cube.copy()
        else:
            cube_fc = cube.copy()
            cube_adi_fc = np.copy(algo_dict['cube_ref'][NAdiRef:NRefT, :, :])
            cube_fc = np.vstack((cube_fc, cube_adi_fc))
            parangles = np.concatenate((angle_list, angle_adi_ref))
        
        cube_fc = cube_inject_companions(
            cube_fc,
            psf_template,
            parangles,
            flux,
            rad_dists=rad_dist,
            theta=br * angle_branch + theta,
            nproc=nproc,
            imlib=imlib,
            interpolation=interpolation,
            verbose=False,
            )
        
        if matrix_adi_ref is not None:
            algo_dict['cube_ref'][NAdiRef:NRefT, :, :] = cube_fc[n:, :, :]
            cube_fc = cube_fc[0:n, :, :]
            
        
        y = cy + rad_dist * \
            np.sin(np.deg2rad(br * angle_branch + theta))
        x = cx + rad_dist * \
            np.cos(np.deg2rad(br * angle_branch + theta))
        fc_map = frame_inject_companion(
            fc_map, psf_template, y, x, flux, imlib, interpolation
        )
        fcy = y
        fcx = x

        if verbose:
            msg2 = "Fake companions injected in branch {} "
            print(msg2.format(br + 1))
            timing(start_time)
    
        frames_fc[:, br, :, :] = algo(cube=cube_fc, angle_list=angle_list, 
                                      fwhm=fwhm_med, verbose=verbose, **algo_dict)
        
        injected_flux = apertureOne_flux(fc_map, fcy, fcx, fwhm_med)
        recovered_flux = np.array([apertureOne_flux(
            (frames_fc[i, br, :, :] - frames_no_fc[i, :, :]), fcy, fcx, fwhm_med
        ) for i in range(0, nnpcs)])
        
        thruput = recovered_flux / injected_flux
        thruput[np.where(thruput < 0)] = 0
        
        Throughput[:, br] = thruput.reshape((nnpcs))
        

    noise_samp = noise
    res_lev_samp = mean_res         
                
    res_lev_samp = np.abs(res_lev_samp)

    noise_samp_sm = noise_samp
    res_lev_samp_sm = res_lev_samp
    
    
    Thru_Cont = np.zeros((nnpcs, 3))
    if isinstance(ncomp, tuple):
        Thru_Cont[:, 2] = ncomp[1]
    else:
        Thru_Cont[:, 2] = ncomp
    
    Thru_Cont[:,0] = [np.nanmean(Throughput[i,:]) for i in range(0, nnpcs)]

    
    if isinstance(starphot, float) or isinstance(starphot, int):
        Thru_Cont[:,1] = (
            (sigma * noise_samp_sm + res_lev_samp_sm) / Thru_Cont[:,0]
        ) / starphot
    else:
        Thru_Cont[:,1] = (
            (sigma * noise_samp_sm + res_lev_samp_sm) / Thru_Cont[:,0]
        ) / np.median(starphot)
        
    return (Thru_Cont, frames_fc)



def contrast_step_dist_opt(
    cube,
    angle_list,
    psf_template,
    fwhm,
    distance,
    pxscale,
    starphot,
    algo,
    step,
    through_thresh=0.1,
    sigma=5,
    nbranch=1,
    theta=0,
    inner_rad=1,
    fc_rad_sep=3,
    noise_sep=1,
    wedge=(0, 360),
    fc_snr=50,
    flux = None,
    student=True,
    transmission=None,
    dpi=vip_figdpi,
    debug=False,
    verbose=True,
    full_output=False,
    save_plot=None,
    object_name=None,
    frame_size=None,
    fix_y_lim=(),
    figsize=vip_figsize,
    algo_class=None,
    matrix_adi_ref=None,
    angle_adi_ref=None,
    **algo_dict,
):
    """
    Estimates the contrast in much the same way contrast_curve does it, but
    only at specific distances provided by the argument 'distance'.
    
    -ncomp: must be list of components that will be tested
    
    -distance:Provides the distances(in fwhm) at which the fake companions will
    be injected. Attention: to limit computational cost, all companions at the 
    different distances will be injected at once. In order to have the most
    accurate estimation of the contrast, these companions should not be in
    the same annulus.
    If distance = 'auto': the distances are automatically calculated to be the
    centers of the annuli in the images if the algorithm used has annuli
    
    -step:in addition to calculating the optimal contrast for each value of ncomp,
    the optimization of the contrast can be done by segmenting the cube in steps.
    The number of images must be divisible by the step
    
    -through_thresh:to select the optimal component, a threshold on the 
    throuput can be used to prevent components with throughputs too low from
    being selected
    """
    if cube.ndim != 3 and cube.ndim != 4:
        raise TypeError("The input array is not a 3d or 4d cube")
    if cube.ndim == 3 and (cube.shape[0] != angle_list.shape[0]):
        raise TypeError("Input parallactic angles vector has wrong length")
    if cube.ndim == 4 and (cube.shape[1] != angle_list.shape[0]):
        raise TypeError("Input parallactic angles vector has wrong length")
    if cube.ndim == 3 and psf_template.ndim != 2:
        raise TypeError("Template PSF is not a frame (for ADI case)")
    if cube.ndim == 4 and psf_template.ndim != 3:
        raise TypeError("Template PSF is not a cube (for ADI+IFS case)")
    if transmission is not None:
        if len(transmission) != 2 and len(transmission) != cube.shape[0] + 1:
            msg = "Wrong shape for transmission should be 2xn_rad or (nch+1) "
            msg += "x n_rad, instead of {}".format(transmission.shape)
            raise TypeError(msg)

    if isinstance(fwhm, (np.ndarray, list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if verbose:
        start_time = time_ini()
        if isinstance(starphot, float) or isinstance(starphot, int):
            msg0 = "ALGO : {}, FWHM = {}, # BRANCHES = {}, SIGMA = {},"
            msg0 += " STARPHOT = {}"
            print(msg0.format(algo.__name__, fwhm_med, nbranch, sigma, starphot))
        else:
            msg0 = "ALGO : {}, FWHM = {}, # BRANCHES = {}, SIGMA = {}"
            print(msg0.format(algo.__name__, fwhm_med, nbranch, sigma))
            
    if cube.ndim != 3 and cube.ndim != 4:
        raise TypeError("The input array is not a 3d or 4d cube")

    else:
        if psf_template.shape[1] % 2 == 0:
            raise ValueError("Only odd-sized PSF is accepted")
        if not hasattr(algo, "__call__"):
            raise TypeError("Parameter `algo` must be a callable function")
        if not isinstance(inner_rad, int):
            raise TypeError("inner_rad must be an integer")
        angular_range = wedge[1] - wedge[0]
        if nbranch > 1 and angular_range < 360:
            msg = "Only a single branch is allowed when working on a wedge"
            raise RuntimeError(msg)
        
    nproc = algo_dict.get("nproc", 1)
    imlib = algo_dict.get("imlib", "vip-fft")
    interpolation = algo_dict.get("interpolation", "lanczos4")
    scaling = algo_dict.get("scaling", None)
    ncomp = algo_dict.get("ncomp")
    
    if cube.ndim == 3:
        if isinstance(ncomp, list):
            nnpcs = len(ncomp)
            if isinstance(ncomp[0], tuple) or isinstance(ncomp[0], np.ndarray):
            #for pca_annular when ncomp is different for each annulus
                nnpcs = len(ncomp[0][0])
        elif isinstance(ncomp, tuple):
            #for ARDI_double_pca function
            if isinstance(ncomp[1], list):
                nnpcs = len(ncomp[1])
            else:
                nnpcs = 1
        else:
            nnpcs = 1
    elif cube.ndim == 4:
        if isinstance(ncomp, list):
            if len(ncomp) == cube.shape[0]:
                nnpcs = 1
            else:
                nnpcs = len(ncomp)
        else:
            nnpcs = 1
            
    algo_name = algo.__name__
    idx = algo.__module__.index('.', algo.__module__.index('.') + 1)
    mod = algo.__module__[:idx]
    tmp = __import__(mod, fromlist=[algo_name.upper()+'_Params'])    
    algo_params = getattr(tmp, algo_name.upper()+'_Params')
    
    algo_supported = ['pca_annular', 'pca_annular_corr', 
                      'pca_annular_multi_epoch', 'pca_annular_corr_multi_epoch']
    if algo_name not in algo_supported:
        raise ValueError("Algorithm is not supported")
        
    SizeImage = int(cube[0].shape[1])
    NbrImages = int(cube.shape[0])
    if algo_name == 'pca_annular_corr':
        epoch_indices =  algo_dict['epoch_indices']
        NbrImages = int(epoch_indices[1]-epoch_indices[0])
    
    if isinstance(step, int):
        if NbrImages % step != 0:
            raise ValueError("Number of images must be divisible by the step")
        step = np.array([step], dtype = int)
        TotalSteps = NbrImages/step
    elif isinstance(step, list) or isinstance(step, np.ndarray):
        step = np.array(step, dtype = int)
        for i in step:
            if NbrImages % i != 0:
                raise ValueError("Number of images must be divisible by all the steps")
        TotalSteps = NbrImages/step
    elif step == 'auto':
        #Find all the divisors of the number of images
        step = []
        lim = int(np.sqrt(NbrImages))
        for i in range(1, lim+1):
            if NbrImages % i == 0:
                step.append(i)
                step.append(NbrImages/i)
        step = np.array(step, dtype = int)
        step = np.sort(step)
        TotalSteps = NbrImages/step
    elif step == 'max':
        step = np.array([NbrImages], dtype = int)
        TotalSteps = NbrImages/step
    else:
        raise ValueError("Step must be an int, list, numpy array or equal to 'auto'")
    NbrStepValue = step.shape[0]
    TotalSteps = TotalSteps.astype(int)
    
    
    frames_fc = np.zeros((nnpcs, nbranch, SizeImage, SizeImage), dtype = float)
    frames_no_fc = np.zeros((nnpcs, SizeImage, SizeImage), dtype = float)
    res_cube_fc = np.zeros((nnpcs, nbranch, NbrImages, SizeImage, SizeImage), dtype = float)
    res_cube_no_fc = np.zeros((nnpcs, NbrImages, SizeImage, SizeImage), dtype = float)
    
    if isinstance(fwhm, (np.ndarray, list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if verbose:
        start_time = time_ini()
        
    
    if matrix_adi_ref is not None:
        if 'cube_ref' in algo_dict.keys() and algo_dict['cube_ref'] is not None:
            NAdiRef = algo_dict['cube_ref'].shape[0]
            algo_dict['cube_ref'] = np.vstack((algo_dict['cube_ref'], matrix_adi_ref))
        else:
            NAdiRef = 0
            algo_dict['cube_ref'] = matrix_adi_ref
        NRefT = algo_dict['cube_ref'].shape[0]
    
    if 'annular' in algo_name:
        if distance == 'auto':
            radius_int = algo_dict['radius_int']
            asize = algo_dict['asize']
            y = cube.shape[2]
            n_annuli = int((y / 2 - radius_int) / asize)
            distance = np.array([radius_int+(asize/2) + i*asize for i in range(0, n_annuli)])/fwhm_med
        elif isinstance(distance, float):
            distance = np.array([distance])
        elif isinstance(distance, list):
            distance = np.array(distance)
        else:
            raise ValueError("distance parameter must be a float, a list or equal to 'auto'")
        
        if algo_name == 'pca_annular' or algo_name == 'pca_annular_corr':
            _, res_cube_no_fc[:, :, :, :], frames_no_fc[:, :, :] = algo(
                        cube=cube, angle_list=angle_list, fwhm=fwhm_med,
                        verbose=verbose, full_output = True, **algo_dict)
        elif algo_name == 'pca_annular_multi_epoch' or algo_name == 'pca_annular_corr_multi_epoch':
            frames_no_fc[:, :, :], res_cube_no_fc[:, :, :, :] = algo(
                        cube=cube, angle_list=angle_list, fwhm=fwhm_med,
                        verbose=verbose, full_output = True, **algo_dict)
    else:
        raise ValueError("Algorithm not supported")
        
        
    rad_dist = distance * fwhm_med
    nbr_dist = distance.shape[0]

    
    #CHANGE NOISE ANNULI TO HAVE IT ONLY HERE AT THIS DISTANCE !!!
    noise = []
    mean_res = []
    noise_avg = np.array([noise_dist(frames_no_fc[n, :, :], rad_dist, fwhm_med, wedge, 
                        False, debug) for n in range(0, nnpcs)])
    
    
    
    
    for i, NbrSteps in enumerate(TotalSteps):
        noise_tmp = []
        mean_res_tmp = []
        for s in range(0, NbrSteps, 1):
            noise_res = np.array(
                [noise_dist(np.median(res_cube_no_fc[n, s*step[i]:(s+1)*step[i]:1, :, :], axis = 0), 
                rad_dist, fwhm_med, wedge, False, debug) for n in range(0, nnpcs)])
            noise_tmp.append(noise_res[:, :, 0])
            mean_res_tmp.append(noise_res[:, :, 1])
        noise.append(noise_tmp)
        mean_res.append(mean_res_tmp)
    
    
    # We crop the PSF and check if PSF has been normalized (so that flux in
    # 1*FWHM aperture = 1) and fix if needed
    new_psf_size = int(round(3 * fwhm_med))
    if new_psf_size % 2 == 0:
        new_psf_size += 1
    
    n, y, x = cube.shape
    psf_template = normalize_psf(
        psf_template,
        fwhm=fwhm,
        verbose=verbose,
        size=min(new_psf_size, psf_template.shape[1]),
    )
    
    
    # Initialize the fake companions
    angle_branch = angular_range / nbranch
    
    Throughput = []
    for NbrSteps in TotalSteps:
        Throughput.append(np.zeros((NbrSteps, nnpcs, nbr_dist, nbranch)))
    
    fc_map = np.zeros((y, x))
    cy, cx = frame_center(cube[0])
    parangles = angle_list

    # each branch is computed separately
    if matrix_adi_ref is not None:
        copy_ref = np.copy(algo_dict['cube_ref'])
        
    loc = np.zeros((nbr_dist, nbranch, 2))
    thruput_avg_tmp = np.zeros((nnpcs, nbr_dist, nbranch))
    all_injected_flux = np.zeros((nbr_dist, nbranch))
    for br in range(nbranch):
        
        if matrix_adi_ref is not None:
            algo_dict['cube_ref'] = np.copy(copy_ref)
        
        # each pattern is computed separately. For each one the companions
        # are separated by "fc_rad_sep * fwhm", interleaving the injections
        fc_map = np.ones_like(cube[0]) * 1e-6
        fcy = 0
        fcx = 0
        
        if flux is None:
            flux = fc_snr * np.array([np.percentile(noise_avg[:, d, 0], 20) for d in range(0,nbr_dist)])
        
        if matrix_adi_ref is None:
            cube_fc = cube.copy()
        else:
            cube_fc = cube.copy()
            cube_adi_fc = np.copy(algo_dict['cube_ref'][NAdiRef:NRefT, :, :])
            cube_fc = np.vstack((cube_fc, cube_adi_fc))
            parangles = np.concatenate((angle_list, angle_adi_ref))
        
        for d in range(0, nbr_dist):
            cube_fc = cube_inject_companions(
                cube_fc,
                psf_template,
                parangles,
                flux[d],
                rad_dists=rad_dist[d],
                theta=br * angle_branch + theta,
                nproc=nproc,
                imlib=imlib,
                interpolation=interpolation,
                verbose=False,
                )
        
        if matrix_adi_ref is not None:
            algo_dict['cube_ref'][NAdiRef:NRefT, :, :] = cube_fc[n:, :, :]
            cube_fc = cube_fc[0:n, :, :]
            
        
        for d in range(0, nbr_dist):
            y = cy + rad_dist[d] * \
                np.sin(np.deg2rad(br * angle_branch + theta))
            x = cx + rad_dist[d] * \
                np.cos(np.deg2rad(br * angle_branch + theta))
            fc_map = frame_inject_companion(
                fc_map, psf_template, y, x, flux[d], imlib, interpolation
            )
            fcy = y
            fcx = x
            loc[d, br ,:] = np.array([fcy, fcx])

        if verbose:
            msg2 = "Fake companions injected in branch {} "
            print(msg2.format(br + 1))
            timing(start_time)
    
        if algo_name == 'pca_annular' or algo_name == 'pca_annular_corr':
            _, res_cube_fc[:, br, : ,:, :], frames_fc[:, br, :, :] = algo(cube=cube_fc, 
                    angle_list=angle_list, fwhm=fwhm_med, verbose=verbose, 
                    full_output = True, **algo_dict)
        elif algo_name == 'pca_annular_multi_epoch' or algo_name == 'pca_annular_corr_multi_epoch':
            frames_fc[:, br, :, :], res_cube_fc[:, br, : ,:, :] = algo(cube=cube_fc, 
                    angle_list=angle_list, fwhm=fwhm_med, verbose=verbose, 
                    full_output = True, **algo_dict)
        

        injected_flux = apertureOne_flux(fc_map, loc[:,br,0], loc[:,br,1], fwhm_med)
        all_injected_flux[:,br] = injected_flux
        recovered_flux_avg = np.array([apertureOne_flux(
            (frames_fc[n, br, :, :] - frames_no_fc[n, :, :]), loc[:,br,0], loc[:,br,1], fwhm_med
        ) for n in range(0, nnpcs)])
        for d in range(0, nbr_dist):
            thruput_avg_tmp[:,d,br] = recovered_flux_avg[:,d]/injected_flux[d]
        thruput_avg_tmp[np.where(thruput_avg_tmp < 0)] = 0
        
        recovered_flux = []
        thruput = []
        for i, NbrSteps in enumerate(TotalSteps):
            recovered_flux_tmp = []
            thruput_tmp = []
            for s in range(0, NbrSteps, 1):
                frame_step_no_fc = np.median(res_cube_no_fc[:, s*step[i]:(s+1)*step[i], :, :], axis = 1)
                frame_step_fc = np.median(res_cube_fc[:, br, s*step[i]:(s+1)*step[i], :, :], axis = 1)
                recovered_flux_tmp.append(np.array([apertureOne_flux(
                    (frame_step_fc[n, :, :] - frame_step_no_fc[n, :, :]), loc[:,br,0], loc[:,br,1], fwhm_med
                ) for n in range(0, nnpcs)]))
                thruput_tmp_tmp = np.copy(recovered_flux_tmp[s])
                for d in range(0, nbr_dist):
                    thruput_tmp_tmp[:,d] = thruput_tmp_tmp[:,d]/injected_flux[d]
                    thruput_tmp.append(np.array(thruput_tmp_tmp))
                thruput_tmp[s][np.where(thruput_tmp[s] < 0)] = 0
                Throughput[i][s,:,:,br] = thruput_tmp[s].reshape((nnpcs, nbr_dist))
            thruput.append(thruput_tmp)
            recovered_flux.append(recovered_flux_tmp)

    noise_samp_sm = noise  
    res_lev_samp_sm = []
    for i in range(0, NbrStepValue):           
        res_lev_samp_sm.append(np.abs(mean_res[i]))
    
    BestNInd = []
    Thru_Cont = []
    for NbrSteps in TotalSteps:
        BestNInd.append(np.zeros((NbrSteps, nbr_dist), dtype=int))
        Thru_Cont.append(np.zeros((NbrSteps, nnpcs, nbr_dist, 2)))
    
    
    for i, NbrSteps in enumerate(TotalSteps):
        for s in range(0, NbrSteps, 1):
            Thru_Cont[i][s,:,:,0] = [np.nanmean(Throughput[i][s,n,:,:], axis = 1) for n in range(0, nnpcs)]
            if isinstance(starphot, float) or isinstance(starphot, int):
                Thru_Cont[i][s,:,:,1] = (
                    (sigma * noise_samp_sm[i][s]) / Thru_Cont[i][s,:,:,0]
                ) / starphot
            else:
                Thru_Cont[i][s,:,:,1] = (
                    (sigma * noise_samp_sm[i][s]) / Thru_Cont[i][s,:,:,0]
                ) / np.median(starphot)


    #SELECT BEST COMP FOR EACH STEP
    #THRESHOLD ON THROUGHPUT + if all contr = infinite at that step, take avg...
    for i, NbrSteps in enumerate(TotalSteps):
        for s in range(0, NbrSteps, 1):
            for d in range(0, nbr_dist):
                indices = np.where(Thru_Cont[i][s,:,d,0] >= through_thresh)[0]
                for n in range(0, nnpcs, 1):
                    if Thru_Cont[i][s, n, d, 1] != np.inf:
                        break
                    BestNInd[i][s,d] = -1
                    continue
                if indices.shape[0] == 0:
                    BestNInd[i][s,d] = -1
                    continue
                Min_ind = indices[np.argmin(Thru_Cont[i][s, indices, d, 1])]
                BestNInd[i][s,d] = Min_ind
    
    
    ncomp = np.array(ncomp)
    BestComp = []
    for i in range(0,NbrStepValue):
        for d in range(0, nbr_dist):
            if np.where(BestNInd[i][:,d] != -1)[0].shape[0] == 0:
                AvgN = 1
            else:
                AvgN = int(np.mean(BestNInd[i][np.where(BestNInd[i][:,d] != -1)[0],d]))
            #AvgN = int(np.nanmean(BestNInd))
            BestNInd[i][np.where(BestNInd[i][:,d] == -1),d] = AvgN
        BestComp.append(ncomp[BestNInd[i]])
        
    
    #Construct final best frames for all the different step value
    final_frame = np.zeros((NbrStepValue, nbr_dist, SizeImage, SizeImage))
    final_frames_br = np.zeros((NbrStepValue, nbr_dist, nbranch, SizeImage, SizeImage))
    for i, NbrSteps in enumerate(TotalSteps):
        final_no_fc = np.zeros((NbrImages, SizeImage, SizeImage), dtype = float)
        final_fc = np.zeros((nbranch, NbrImages, SizeImage, SizeImage), dtype = float)
        for d in range(0, nbr_dist):
            for s in range(0, NbrSteps, 1):
                final_no_fc[s*step[i]:(s+1)*step[i],:,:] = res_cube_no_fc[BestNInd[i][s,d],s*step[i]:(s+1)*step[i],:,:]
                for br in range(0, nbranch):
                    final_fc[br,s*step[i]:(s+1)*step[i],:,:] = res_cube_fc[BestNInd[i][s,d],br,s*step[i]:(s+1)*step[i],:,:]
    
            final_frame[i,d,:,:] = np.median(final_no_fc[:,:,:], axis = 0)
            final_frames_br[i,d,:,:,:] = np.array([np.median(final_fc[br,:,:,:], axis = 0) 
                                            for br in range(0, nbranch)])
    
    final_noise_res = np.zeros((NbrStepValue, nbr_dist, 2))
    for i in range(0,NbrStepValue):
        final_noise_res[i,:,:] = np.array([noise_dist(final_frame[i,d], 
                rad_dist[d], fwhm_med, wedge, False, debug) for d in range(0, nbr_dist)]).reshape(nbr_dist, 2)
    final_noise = final_noise_res[:, :, 0]
    final_mean_res = np.abs(final_noise_res[:, :, 1])
    
    final_recovered_fluxes = np.zeros((NbrStepValue, nbr_dist, nbranch))
    for i in range(0,NbrStepValue):
        for d in range(0, nbr_dist):
            final_recovered_fluxes[i,d,:] = np.array([apertureOne_flux(
                (final_frames_br[i,d,br,:,:] - final_frame[i,d,:,:]), loc[d,br,0], loc[d,br,1], fwhm_med) 
                                            for br in range(0, nbranch)]).reshape(nbranch)
    
    
    final_thruput = np.zeros_like(final_recovered_fluxes)
    for d in range(0, nbr_dist):
        for br in range(nbranch):
            final_thruput[:,d,br] = final_recovered_fluxes[:,d,br] / all_injected_flux[d, br]
    final_thruput[np.where(final_thruput < 0)] = 0
    final_result = np.zeros((NbrStepValue, nbr_dist, 2))
    for i in range(0,NbrStepValue):
        final_result[i,:,0] = np.nanmean(final_thruput[i,:,:], axis = 1)
        if isinstance(starphot, float) or isinstance(starphot, int):
            final_result[i,:,1] = (
                (sigma * final_noise[i,:]) / final_result[i,:,0]
            ) / starphot
        else:
            final_result[i,:,1] = (
                (sigma * final_noise[i,:]) / final_result[i,:,0]
            ) / np.median(starphot)
            
    thruput_avg = np.nanmean(thruput_avg_tmp[:,:,:], axis = 2)
    thru_cont_avg = np.zeros((nnpcs, nbr_dist, 3))
    thru_cont_avg[:,:,0] = thruput_avg
    
    print(thruput_avg)
    print(noise_avg)
    if isinstance(starphot, float) or isinstance(starphot, int):
        thru_cont_avg[:,:,1] = (
            (sigma * noise_avg[:,:,0]) / thru_cont_avg[:,:,0]
        ) / starphot
    else:
        thru_cont_avg[:,:,1] = (
            (sigma * noise_avg[:,:,0]) / thru_cont_avg[:,:,0]
        ) / np.median(starphot)
        
    return (thru_cont_avg, final_result, rad_dist, step, BestComp, final_frame, final_frames_br)



def contrast_multi_epoch_opt(
    cube,
    angle_list,
    psf_template,
    fwhm,
    distance,
    pxscale,
    starphot,
    algo,
    nbr_epochs,
    step,
    through_thresh=0.1,
    sigma=5,
    nbranch=1,
    theta=0,
    inner_rad=1,
    fc_rad_sep=3,
    noise_sep=1,
    wedge=(0, 360),
    fc_snr=50,
    flux = None,
    cube_delimiter=None,
    cube_ref_delimiter=None,
    epoch_indices=None,
    student=True,
    transmission=None,
    dpi=vip_figdpi,
    debug=False,
    verbose=True,
    full_output=False,
    save_plot=None,
    object_name=None,
    frame_size=None,
    fix_y_lim=(),
    figsize=vip_figsize,
    algo_class=None,
    matrix_adi_ref=None,
    angle_adi_ref=None,
    **algo_dict,
    ):

    results = []
    algo_dict_copy = algo_dict.copy()
    
    if isinstance(cube_delimiter, list):
        cube_delimiter = np.array(cube_delimiter)
    if cube_delimiter.shape[0] == nbr_epochs*2:
        R = int(1)
    else:
        R = int(0)
    
    for N in range(0, nbr_epochs):
        algo_dict = algo_dict_copy.copy()
        cube_adi = cube[cube_delimiter[N+R*N]:cube_delimiter[N+R*N+1]]
        this_angle_list = angle_list[cube_delimiter[N+R*N]:cube_delimiter[N+R*N+1]]
        if cube_ref_delimiter is not None:
            cube_ref_delimiter = np.array(cube_ref_delimiter)
            Rr = int(0)
            if cube_ref_delimiter.shape[0] == nbr_epochs*2:
                Rr = int(1)
            algo_dict['cube_ref'] = algo_dict['cube_ref'][cube_ref_delimiter[N+Rr*N]:
                                             cube_ref_delimiter[N+Rr*N+1],:,:]
                
        if algo.__name__ == 'pca_annular_corr':
            if epoch_indices is not None:
                epoch_indices = np.array(epoch_indices)
                Re = int(0)
                if epoch_indices.shape[0] == nbr_epochs*2:
                    Re = int(1)
                algo_dict['epoch_indices'] = epoch_indices[N+Re*N:N+Re*N+2]
            else:
                algo_dict['epoch_indices'] = (cube_delimiter[N+R*N],cube_delimiter[N+R*N+1])
            
        if 'delta_rot' in algo_dict.keys():
            if isinstance(algo_dict['delta_rot'], list):
                algo_dict['delta_rot'] = np.array(algo_dict['delta_rot'])
            if isinstance(algo_dict['delta_rot'], np.ndarray):
                if algo_dict['delta_rot'].shape[0] != nbr_epochs:
                    raise ValueError('delta_rot has wrong length')
                algo_dict['delta_rot'] = algo_dict['delta_rot'][N]
        
        if step == 'max':
            this_step = cube_adi.shape[0]
            if epoch_indices is not None:
                this_step = int(algo_dict['epoch_indices'][1]-algo_dict['epoch_indices'][0])
        else:
            this_step = step
        
        try:
            res = contrast_step_dist_opt(
                cube_adi,
                this_angle_list,
                psf_template,
                fwhm,
                distance,
                pxscale,
                starphot,
                algo,
                this_step,
                through_thresh,
                sigma,
                nbranch,
                theta,
                inner_rad,
                fc_rad_sep,
                noise_sep,
                wedge,
                fc_snr,
                flux,
                student,
                transmission,
                dpi,
                debug,
                verbose,
                full_output,
                save_plot,
                object_name,
                frame_size,
                fix_y_lim,
                figsize,
                algo_class,
                matrix_adi_ref,
                angle_adi_ref,
                **algo_dict,
            )
            results.append(res)
            
        except Exception as e:
            if hasattr(e, 'message'):
                print(e.message)
                results.append(e.message)
            else:
                print(e)
                results.append(e)
            
        if verbose:
            print(results[N])
            
    return results



def contrast_step_dist(
    cube,
    angle_list,
    psf_template,
    fwhm,
    distance,
    pxscale,
    starphot,
    algo,
    through_thresh=0.1,
    sigma=5,
    nbranch=1,
    theta=0,
    inner_rad=1,
    fc_rad_sep=3,
    noise_sep=1,
    wedge=(0, 360),
    fc_snr=50,
    flux = None,
    student=True,
    transmission=None,
    dpi=vip_figdpi,
    debug=False,
    verbose=True,
    full_output=False,
    save_plot=None,
    object_name=None,
    frame_size=None,
    fix_y_lim=(),
    figsize=vip_figsize,
    algo_class=None,
    matrix_adi_ref=None,
    angle_adi_ref=None,
    **algo_dict,
):
    """
    Estimates the contrast in much the same way contrast_curve does it, but
    only at specific distances provided by the argument 'distance'.
    
    -ncomp: must be list of components that will be tested
    
    -distance:Provides the distances(in fwhm) at which the fake companions will
    be injected. Attention: to limit computational cost, all companions at the 
    different distances will be injected at once. In order to have the most
    accurate estimation of the contrast, these companions should not be in
    the same annulus.
    If distance = 'auto': the distances are automatically calculated to be the
    centers of the annuli in the images if the algorithm used has annuli
    
    -step:in addition to calculating the optimal contrast for each value of ncomp,
    the optimization of the contrast can be done by segmenting the cube in steps.
    The number of images must be divisible by the step
    
    -through_thresh:to select the optimal component, a threshold on the 
    throuput can be used to prevent components with throughputs too low from
    being selected
    """
    if cube.ndim != 3 and cube.ndim != 4:
        raise TypeError("The input array is not a 3d or 4d cube")
    if cube.ndim == 3 and (cube.shape[0] != angle_list.shape[0]):
        raise TypeError("Input parallactic angles vector has wrong length")
    if cube.ndim == 4 and (cube.shape[1] != angle_list.shape[0]):
        raise TypeError("Input parallactic angles vector has wrong length")
    if cube.ndim == 3 and psf_template.ndim != 2:
        raise TypeError("Template PSF is not a frame (for ADI case)")
    if cube.ndim == 4 and psf_template.ndim != 3:
        raise TypeError("Template PSF is not a cube (for ADI+IFS case)")
    if transmission is not None:
        if len(transmission) != 2 and len(transmission) != cube.shape[0] + 1:
            msg = "Wrong shape for transmission should be 2xn_rad or (nch+1) "
            msg += "x n_rad, instead of {}".format(transmission.shape)
            raise TypeError(msg)

    if isinstance(fwhm, (np.ndarray, list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if verbose:
        start_time = time_ini()
        if isinstance(starphot, float) or isinstance(starphot, int):
            msg0 = "ALGO : {}, FWHM = {}, # BRANCHES = {}, SIGMA = {},"
            msg0 += " STARPHOT = {}"
            print(msg0.format(algo.__name__, fwhm_med, nbranch, sigma, starphot))
        else:
            msg0 = "ALGO : {}, FWHM = {}, # BRANCHES = {}, SIGMA = {}"
            print(msg0.format(algo.__name__, fwhm_med, nbranch, sigma))
            
    if cube.ndim != 3 and cube.ndim != 4:
        raise TypeError("The input array is not a 3d or 4d cube")

    else:
        if psf_template.shape[1] % 2 == 0:
            raise ValueError("Only odd-sized PSF is accepted")
        if not hasattr(algo, "__call__"):
            raise TypeError("Parameter `algo` must be a callable function")
        if not isinstance(inner_rad, int):
            raise TypeError("inner_rad must be an integer")
        angular_range = wedge[1] - wedge[0]
        if nbranch > 1 and angular_range < 360:
            msg = "Only a single branch is allowed when working on a wedge"
            raise RuntimeError(msg)
        
    nproc = algo_dict.get("nproc", 1)
    imlib = algo_dict.get("imlib", "vip-fft")
    interpolation = algo_dict.get("interpolation", "lanczos4")
    scaling = algo_dict.get("scaling", None)
    ncomp = algo_dict.get("ncomp")
    
    if cube.ndim == 3:
        if isinstance(ncomp, list):
            nnpcs = len(ncomp)
            if isinstance(ncomp[0], tuple) or isinstance(ncomp[0], np.ndarray):
            #for pca_annular when ncomp is different for each annulus
                nnpcs = len(ncomp[0][0])
        elif isinstance(ncomp, tuple):
            #for ARDI_double_pca function
            if isinstance(ncomp[1], list):
                nnpcs = len(ncomp[1])
            else:
                nnpcs = 1
        else:
            nnpcs = 1
    elif cube.ndim == 4:
        if isinstance(ncomp, list):
            if len(ncomp) == cube.shape[0]:
                nnpcs = 1
            else:
                nnpcs = len(ncomp)
        else:
            nnpcs = 1
    
    if np.isscalar(ncomp):
        ncomp = np.array([ncomp])
    elif isinstance(ncomp, list):
        ncomp = np.array(ncomp)
            
    algo_name = algo.__name__
    idx = algo.__module__.index('.', algo.__module__.index('.') + 1)
    mod = algo.__module__[:idx]
    tmp = __import__(mod, fromlist=[algo_name.upper()+'_Params'])    
    algo_params = getattr(tmp, algo_name.upper()+'_Params')
    
    algo_supported = ['pca_annular', 'pca_annular_corr', 
                      'pca_annular_multi_epoch', 'pca_annular_corr_multi_epoch']
    if algo_name not in algo_supported:
        raise ValueError("Algorithm is not supported")
        
    SizeImage = int(cube[0].shape[1])
    NbrImages = int(cube.shape[0])
    if algo_name == 'pca_annular_corr':
        epoch_indices =  algo_dict['epoch_indices']
        NbrImages = int(epoch_indices[1]-epoch_indices[0])
    
    
    frames_fc = np.zeros((nnpcs, nbranch, SizeImage, SizeImage), dtype = float)
    frames_no_fc = np.zeros((nnpcs, SizeImage, SizeImage), dtype = float)
    res_cube_fc = np.zeros((nnpcs, nbranch, NbrImages, SizeImage, SizeImage), dtype = float)
    res_cube_no_fc = np.zeros((nnpcs, NbrImages, SizeImage, SizeImage), dtype = float)
    
    if isinstance(fwhm, (np.ndarray, list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if verbose:
        start_time = time_ini()
        
    
    if matrix_adi_ref is not None:
        if 'cube_ref' in algo_dict.keys() and algo_dict['cube_ref'] is not None:
            NAdiRef = algo_dict['cube_ref'].shape[0]
            algo_dict['cube_ref'] = np.vstack((algo_dict['cube_ref'], matrix_adi_ref))
        else:
            NAdiRef = 0
            algo_dict['cube_ref'] = matrix_adi_ref
        NRefT = algo_dict['cube_ref'].shape[0]
    
    if 'annular' in algo_name:
        if distance == 'auto':
            radius_int = algo_dict['radius_int']
            asize = algo_dict['asize']
            y = cube.shape[2]
            n_annuli = int((y / 2 - radius_int) / asize)
            distance = np.array([radius_int+(asize/2) + i*asize for i in range(0, n_annuli)])/fwhm_med
        elif isinstance(distance, float):
            distance = np.array([distance])
        elif isinstance(distance, list):
            distance = np.array(distance)
        else:
            raise ValueError("distance parameter must be a float, a list or equal to 'auto'")
        
        if algo_name == 'pca_annular' or algo_name == 'pca_annular_corr':
            _, res_cube_no_fc[:, :, :, :], frames_no_fc[:, :, :] = algo(
                        cube=cube, angle_list=angle_list, fwhm=fwhm_med,
                        verbose=verbose, full_output = True, **algo_dict)
        elif algo_name == 'pca_annular_multi_epoch' or algo_name == 'pca_annular_corr_multi_epoch':
            frames_no_fc[:, :, :], res_cube_no_fc[:, :, :, :] = algo(
                        cube=cube, angle_list=angle_list, fwhm=fwhm_med,
                        verbose=verbose, full_output = True, **algo_dict)
    else:
        raise ValueError("Algorithm not supported")
        
        
    rad_dist = distance * fwhm_med
    nbr_dist = distance.shape[0]

    
    noise_avg = np.array([noise_dist(frames_no_fc[n, :, :], rad_dist, fwhm_med, wedge, 
                        False, debug) for n in range(0, nnpcs)])
    
    noise = noise_avg[:, :, 0]
    mean_res = noise_avg[:, :, 1]
    
    
    # We crop the PSF and check if PSF has been normalized (so that flux in
    # 1*FWHM aperture = 1) and fix if needed
    new_psf_size = int(round(3 * fwhm_med))
    if new_psf_size % 2 == 0:
        new_psf_size += 1
    
    n, y, x = cube.shape
    psf_template = normalize_psf(
        psf_template,
        fwhm=fwhm,
        verbose=verbose,
        size=min(new_psf_size, psf_template.shape[1]),
    )
    
    
    # Initialize the fake companions
    angle_branch = angular_range / nbranch
    
    Throughput = np.zeros((nnpcs, nbr_dist, nbranch))
    
    fc_map = np.zeros((y, x))
    cy, cx = frame_center(cube[0])
    parangles = angle_list

    # each branch is computed separately
    if matrix_adi_ref is not None:
        copy_ref = np.copy(algo_dict['cube_ref'])
        
    loc = np.zeros((nbr_dist, nbranch, 2))
    thruput = np.zeros((nnpcs, nbr_dist, nbranch))
    recovered_flux = np.zeros((nnpcs, nbr_dist, nbranch))
    all_injected_flux = np.zeros((nbr_dist, nbranch))
    
    
    for n in range(0, nnpcs):
        for br in range(nbranch):
        
            if matrix_adi_ref is not None:
                algo_dict['cube_ref'] = np.copy(copy_ref)
        
            # each pattern is computed separately. For each one the companions
            # are separated by "fc_rad_sep * fwhm", interleaving the injections
            fc_map = np.ones_like(cube[0]) * 1e-6
            fcy = 0
            fcx = 0
            if flux is None:
                flux = fc_snr * np.array(noise_avg[n, :, 0])
            print(flux)
        
            if matrix_adi_ref is None:
                cube_fc = cube.copy()
            else:
                cube_fc = cube.copy()
                cube_adi_fc = np.copy(algo_dict['cube_ref'][NAdiRef:NRefT, :, :])
                cube_fc = np.vstack((cube_fc, cube_adi_fc))
                parangles = np.concatenate((angle_list, angle_adi_ref))
        
            for d in range(0, nbr_dist):
                cube_fc = cube_inject_companions(
                    cube_fc,
                    psf_template,
                    parangles,
                    flux[d],
                    rad_dists=rad_dist[d],
                    theta=br * angle_branch + theta,
                    nproc=nproc,
                    imlib=imlib,
                    interpolation=interpolation,
                    verbose=False,
                    )
        
            if matrix_adi_ref is not None:
                algo_dict['cube_ref'][NAdiRef:NRefT, :, :] = cube_fc[n:, :, :]
                cube_fc = cube_fc[0:n, :, :]
            
        
            for d in range(0, nbr_dist):
                y = cy + rad_dist[d] * \
                    np.sin(np.deg2rad(br * angle_branch + theta))
                x = cx + rad_dist[d] * \
                    np.cos(np.deg2rad(br * angle_branch + theta))
                fc_map = frame_inject_companion(
                    fc_map, psf_template, y, x, flux[d], imlib, interpolation
                    )
                fcy = y
                fcx = x
                loc[d, br ,:] = np.array([fcy, fcx])

            if verbose:
                msg2 = "Fake companions injected in branch {} "
                print(msg2.format(br + 1))
                timing(start_time)
    
            if algo_name == 'pca_annular' or algo_name == 'pca_annular_corr':
                if not isinstance(algo_dict['ncomp'], tuple):
                    algo_dict['ncomp'] = ncomp[n]
                _, res_cube_fc[n, br, : ,:, :], frames_fc[n, br, :, :] = algo(cube=cube_fc, 
                    angle_list=angle_list, fwhm=fwhm_med, verbose=verbose, 
                    full_output = True, **algo_dict)
            elif algo_name == 'pca_annular_multi_epoch' or algo_name == 'pca_annular_corr_multi_epoch':
                frames_fc[n, br, :, :], res_cube_fc[n, br, : ,:, :] = algo(cube=cube_fc, 
                    angle_list=angle_list, fwhm=fwhm_med, verbose=verbose, 
                    full_output = True, **algo_dict)
        

            injected_flux = apertureOne_flux(fc_map, loc[:,br,0], loc[:,br,1], fwhm_med)
            all_injected_flux[:,br] = injected_flux
            recovered_flux[n,:,br] = apertureOne_flux(
                (frames_fc[n, br, :, :] - frames_no_fc[n, :, :]), loc[:,br,0], loc[:,br,1], fwhm_med)
                
            for d in range(0, nbr_dist):
                thruput[n,d,br] = recovered_flux[n,d,br]/injected_flux[d]
            thruput[np.where(thruput < 0)] = 0
            
    thruput = np.nanmean(thruput[:,:,:], axis = 2)
    thru_cont_avg = np.zeros((nnpcs, nbr_dist, 3))
    thru_cont_avg[:,:,0] = thruput
    if isinstance(starphot, float) or isinstance(starphot, int):
        thru_cont_avg[:,:,1] = (
            (sigma * noise_avg[:,:,0]) / thru_cont_avg[:,:,0]
        ) / starphot
    else:
        thru_cont_avg[:,:,1] = (
            (sigma * noise_avg[:,:,0]) / thru_cont_avg[:,:,0]
        ) / np.median(starphot)
    if 'multi_epoch' not in algo_name and not isinstance(ncomp, tuple) and ncomp.shape[0] != 1:
        thru_cont_avg[:,:,2] = ncomp.reshape(ncomp.shape[0],1)
        
    print(noise_avg)
    print(nnpcs)
    print(thru_cont_avg[:,:,0])
    print(starphot)
        
        
    if student:
        n_res_els = np.floor(rad_dist / fwhm_med * 2 * np.pi)
        ss_corr = np.sqrt(1 + 1 / n_res_els)
        sigma_corr = stats.t.ppf(stats.norm.cdf(sigma), n_res_els - 1) * ss_corr
        if isinstance(starphot, float) or isinstance(starphot, int):
            Student_res = (
                (sigma_corr * noise_avg[:,:,0]) / thru_cont_avg[:,:,0]
            ) / starphot
        else:
            Student_res = (
                (sigma_corr * noise_avg[:,:,0]) / thru_cont_avg[:,:,0]
            ) / np.median(starphot)
        Student_res[np.where(Student_res < 0)] = 1
        Student_res[np.where(Student_res > 1)] = 1
        return (thru_cont_avg, Student_res, rad_dist)
    else:
        return (thru_cont_avg, rad_dist)



def contrast_multi_epoch(
    cube,
    angle_list,
    psf_template,
    fwhm,
    distance,
    pxscale,
    starphot,
    algo,
    nbr_epochs,
    through_thresh=0.1,
    sigma=5,
    nbranch=1,
    theta=0,
    inner_rad=1,
    fc_rad_sep=3,
    noise_sep=1,
    wedge=(0, 360),
    fc_snr=50,
    flux = None,
    cube_delimiter=None,
    cube_ref_delimiter=None,
    epoch_indices=None,
    student=True,
    transmission=None,
    dpi=vip_figdpi,
    debug=False,
    verbose=True,
    full_output=False,
    save_plot=None,
    object_name=None,
    frame_size=None,
    fix_y_lim=(),
    figsize=vip_figsize,
    algo_class=None,
    matrix_adi_ref=None,
    angle_adi_ref=None,
    **algo_dict,
    ):

    results = []
    algo_dict_copy = algo_dict.copy()
    
    if isinstance(cube_delimiter, list):
        cube_delimiter = np.array(cube_delimiter)
    if cube_delimiter.shape[0] == nbr_epochs*2:
        R = int(1)
    else:
        R = int(0)
    
    for N in range(0, nbr_epochs):
        algo_dict = algo_dict_copy.copy()
        cube_adi = cube[cube_delimiter[N+R*N]:cube_delimiter[N+R*N+1]]
        this_angle_list = angle_list[cube_delimiter[N+R*N]:cube_delimiter[N+R*N+1]]
        if cube_ref_delimiter is not None:
            cube_ref_delimiter = np.array(cube_ref_delimiter)
            Rr = int(0)
            if cube_ref_delimiter.shape[0] == nbr_epochs*2:
                Rr = int(1)
            algo_dict['cube_ref'] = algo_dict['cube_ref'][cube_ref_delimiter[N+Rr*N]:
                                             cube_ref_delimiter[N+Rr*N+1],:,:]
                
        if algo.__name__ == 'pca_annular_corr':
            if epoch_indices is not None:
                epoch_indices = np.array(epoch_indices)
                Re = int(0)
                if epoch_indices.shape[0] == nbr_epochs*2:
                    Re = int(1)
                algo_dict['epoch_indices'] = epoch_indices[N+Re*N:N+Re*N+2]
            else:
                algo_dict['epoch_indices'] = (cube_delimiter[N+R*N],cube_delimiter[N+R*N+1])
            
        if 'delta_rot' in algo_dict.keys():
            if isinstance(algo_dict['delta_rot'], list):
                algo_dict['delta_rot'] = np.array(algo_dict['delta_rot'])
            if isinstance(algo_dict['delta_rot'], np.ndarray):
                if algo_dict['delta_rot'].shape[0] != nbr_epochs:
                    raise ValueError('delta_rot has wrong length')
                algo_dict['delta_rot'] = algo_dict['delta_rot'][N]
        
        
        try:
            res = contrast_step_dist(
                cube_adi,
                this_angle_list,
                psf_template,
                fwhm,
                distance,
                pxscale,
                starphot,
                algo,
                through_thresh,
                sigma,
                nbranch,
                theta,
                inner_rad,
                fc_rad_sep,
                noise_sep,
                wedge,
                fc_snr,
                flux,
                student,
                transmission,
                dpi,
                debug,
                verbose,
                full_output,
                save_plot,
                object_name,
                frame_size,
                fix_y_lim,
                figsize,
                algo_class,
                matrix_adi_ref,
                angle_adi_ref,
                **algo_dict,
            )
            results.append(res)
            
        except Exception as e:
            if hasattr(e, 'message'):
                print(e.message)
                results.append(e.message)
            else:
                print(e)
                results.append(e)
            
        if verbose:
            print(results[N])
            
    return results



def contrast_multi_epoch_walk(
    cube,
    angle_list,
    psf_template,
    fwhm,
    distance,
    pxscale,
    starphot,
    algo,
    nbr_epochs,
    iterations=5,
    through_thresh=0.1,
    snr_thresh=0,
    sigma=5,
    nbranch=1,
    theta=0,
    inner_rad=1,
    fc_rad_sep=3,
    approximation = 0,
    switch = 3,
    flux_increase = False,
    noise_sep=1,
    wedge=(0, 360),
    fc_snr=50,
    flux = None,
    cube_delimiter=None,
    cube_ref_delimiter=None,
    epoch_indices=None,
    student=True,
    transmission=None,
    dpi=vip_figdpi,
    debug=False,
    verbose=True,
    full_output=False,
    save_plot=None,
    object_name=None,
    frame_size=None,
    fix_y_lim=(),
    figsize=vip_figsize,
    algo_class=None,
    matrix_adi_ref=None,
    angle_adi_ref=None,
    source_xy = None,
    exclude_negative_lobes=False,
    fmerit = 'mean',
    **algo_dict,
):
    
    from ..metrics import snr
    def get_snr(frame, y, x, fwhm, fmerit):
        """
        """
        if fmerit == 'max':
            yy, xx = disk((y, x), fwhm / 2.)
            res = [snr(frame, (x_, y_), fwhm, plot=False, verbose=False,
                       exclude_negative_lobes=exclude_negative_lobes,
                       full_output=True)
                   for y_, x_ in zip(yy, xx)]
            snr_pixels = np.array(res, dtype=object)[:, -1]
            fluxes = np.array(res, dtype=object)[:, 2]
            argm = np.argmax(snr_pixels)
            # integrated fluxes for the max snr
            return np.max(snr_pixels), fluxes[argm]

        elif fmerit == 'px':
            res = snr(frame, (x, y), fwhm, plot=False, verbose=False,
                      exclude_negative_lobes=exclude_negative_lobes,
                      full_output=True)
            snrpx = res[-1]
            fluxpx = np.array(res, dtype=object)[2]
            # integrated fluxes for the given px
            return snrpx, fluxpx

        elif fmerit == 'mean':
            yy, xx = disk((y, x), fwhm / 2.)
            res = [snr(frame, (x_, y_), fwhm, plot=False, verbose=False,
                       exclude_negative_lobes=exclude_negative_lobes,
                       full_output=True) for y_, x_
                   in zip(yy, xx)]
            snr_pixels = np.array(res, dtype=object)[:, -1]
            fluxes = np.array(res, dtype=object)[:, 2]
            # mean of the integrated fluxes (shifting the aperture)
            return np.mean(snr_pixels), np.mean(fluxes)
        
    #-----------------------------------------------------------------

    if cube.ndim != 3 and cube.ndim != 4:
        raise TypeError("The input array is not a 3d or 4d cube")
    if cube.ndim == 3 and (cube.shape[0] != angle_list.shape[0]):
        raise TypeError("Input parallactic angles vector has wrong length")
    if cube.ndim == 4 and (cube.shape[1] != angle_list.shape[0]):
        raise TypeError("Input parallactic angles vector has wrong length")
    if cube.ndim == 3 and psf_template.ndim != 2:
        raise TypeError("Template PSF is not a frame (for ADI case)")
    if cube.ndim == 4 and psf_template.ndim != 3:
        raise TypeError("Template PSF is not a cube (for ADI+IFS case)")
    if transmission is not None:
        if len(transmission) != 2 and len(transmission) != cube.shape[0] + 1:
            msg = "Wrong shape for transmission should be 2xn_rad or (nch+1) "
            msg += "x n_rad, instead of {}".format(transmission.shape)
            raise TypeError(msg)

    if isinstance(fwhm, (np.ndarray, list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if verbose:
        start_time = time_ini()
        if isinstance(starphot, float) or isinstance(starphot, int):
            msg0 = "ALGO : {}, FWHM = {}, # BRANCHES = {}, SIGMA = {},"
            msg0 += " STARPHOT = {}"
            print(msg0.format(algo.__name__, fwhm_med, nbranch, sigma, starphot))
        else:
            msg0 = "ALGO : {}, FWHM = {}, # BRANCHES = {}, SIGMA = {}"
            print(msg0.format(algo.__name__, fwhm_med, nbranch, sigma))
            
    if cube.ndim != 3 and cube.ndim != 4:
        raise TypeError("The input array is not a 3d or 4d cube")

    else:
        if psf_template.shape[1] % 2 == 0:
            raise ValueError("Only odd-sized PSF is accepted")
        if not hasattr(algo, "__call__"):
            raise TypeError("Parameter `algo` must be a callable function")
        if not isinstance(inner_rad, int):
            raise TypeError("inner_rad must be an integer")
        angular_range = wedge[1] - wedge[0]
        if nbranch > 1 and angular_range < 360:
            msg = "Only a single branch is allowed when working on a wedge"
            raise RuntimeError(msg)
        
    nproc = algo_dict.get("nproc", 1)
    imlib = algo_dict.get("imlib", "vip-fft")
    interpolation = algo_dict.get("interpolation", "lanczos4")
    scaling = algo_dict.get("scaling", None)
    ncomp = algo_dict.get("ncomp")
    
    nnpcs = len(ncomp)
    
    if np.isscalar(ncomp):
        ncomp = np.array([ncomp])
    elif isinstance(ncomp, list):
        ncomp = np.array(ncomp)
            
    algo_name = algo.__name__
    idx = algo.__module__.index('.', algo.__module__.index('.') + 1)
    mod = algo.__module__[:idx]
    tmp = __import__(mod, fromlist=[algo_name.upper()+'_Params'])    
    algo_params = getattr(tmp, algo_name.upper()+'_Params')
    
    algo_supported = ['pca_annular', 'pca_annular_corr', 
                      'pca_annular_multi_epoch', 'pca_annular_corr_multi_epoch']
    if algo_name not in algo_supported:
        raise ValueError("Algorithm is not supported")
    
    if isinstance(fwhm, (np.ndarray, list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if verbose:
        start_time = time_ini()
        
    
    if matrix_adi_ref is not None:
        if 'cube_ref' in algo_dict.keys() and algo_dict['cube_ref'] is not None:
            NAdiRef = algo_dict['cube_ref'].shape[0]
            algo_dict['cube_ref'] = np.vstack((algo_dict['cube_ref'], matrix_adi_ref))
        else:
            NAdiRef = 0
            algo_dict['cube_ref'] = matrix_adi_ref
        NRefT = algo_dict['cube_ref'].shape[0]
    
    if 'annular' in algo_name:
        if distance == 'auto':
            radius_int = algo_dict['radius_int']
            asize = algo_dict['asize']
            y = cube.shape[2]
            n_annuli = int((y / 2 - radius_int) / asize)
            distance = np.array([radius_int+(asize/2) + i*asize for i in range(0, n_annuli)])/fwhm_med
        elif isinstance(distance, float):
            distance = np.array([distance])
        elif isinstance(distance, list):
            distance = np.array(distance)
        else:
            raise ValueError("distance parameter must be a float, a list or equal to 'auto'")
            
    SizeImage = int(cube[0].shape[1])
    NbrImages = int(cube.shape[0])
    
    frames_basis_fc = np.zeros((nnpcs, nbranch, SizeImage, SizeImage), dtype = float)
    frames_basis_no_fc = np.zeros((nnpcs, SizeImage, SizeImage), dtype = float)
    res_cube_fc = np.zeros((nnpcs, nbranch, NbrImages, SizeImage, SizeImage), dtype = float)
    res_cube_no_fc = np.zeros((nnpcs, NbrImages, SizeImage, SizeImage), dtype = float)    
    
    rad_dist = distance * fwhm_med
    nbr_dist = distance.shape[0]
    
    results = []
    algo_dict_copy = algo_dict.copy()
    
    if isinstance(cube_delimiter, list):
        cube_delimiter = np.array(cube_delimiter)
    if cube_delimiter.shape[0] == nbr_epochs*2:
        R = int(1)
    else:
        R = int(0)

    indices_epochs = np.zeros(nbr_epochs*2, dtype = int)
    for N in range(nbr_epochs):
        algo_dict = algo_dict_copy.copy()
        
        if len(epoch_indices) == 2 and nbr_epochs > 1:
            algo_dict['epoch_indices'] = epoch_indices
            _, res_cube_no_fc, _ = algo(
                cube=cube, angle_list=angle_list, fwhm=fwhm_med,
                verbose=verbose, full_output = True, **algo_dict)
            step = int(cube.shape[0]/nbr_epochs)
            epoch_indices = np.arange(0, cube.shape[0]+1, int(cube.shape[0]/nbr_epochs))
            indices_epochs = [epoch_indices[int(j/2) + j%2] for j in range(0, nbr_epochs*2)]
            break
        
        indices_cube_adi = (cube_delimiter[N+R*N], cube_delimiter[N+R*N+1])
        cube_adi = cube[indices_cube_adi[0]:indices_cube_adi[1]]
        this_angle_list = angle_list[cube_delimiter[N+R*N]:cube_delimiter[N+R*N+1]]
        if cube_ref_delimiter is not None:
            cube_ref_delimiter = np.array(cube_ref_delimiter)
            Rr = int(0)
            if cube_ref_delimiter.shape[0] == nbr_epochs*2:
                Rr = int(1)
            indices_cube_rdi = (cube_ref_delimiter[N+Rr*N],cube_ref_delimiter[N+Rr*N+1])
            algo_dict['cube_ref'] = algo_dict['cube_ref'][indices_cube_rdi[0]:
                                             indices_cube_rdi[1],:,:]
                
        if algo.__name__ == 'pca_annular_corr':
            if epoch_indices is not None:
                epoch_indices = np.array(epoch_indices)
                Re = int(0)
                if epoch_indices.shape[0] == nbr_epochs*2:
                    Re = int(1)
                algo_dict['epoch_indices'] = epoch_indices[N+Re*N:N+Re*N+2]
            else:
                algo_dict['epoch_indices'] = (indices_cube_adi[0],indices_cube_adi[1])
            
        if 'delta_rot' in algo_dict.keys():
            if isinstance(algo_dict['delta_rot'], list):
                algo_dict['delta_rot'] = np.array(algo_dict['delta_rot'])
            if isinstance(algo_dict['delta_rot'], np.ndarray):
                if algo_dict['delta_rot'].shape[0] != nbr_epochs:
                    raise ValueError('delta_rot has wrong length')
                algo_dict['delta_rot'] = algo_dict['delta_rot'][N]
        
        
        if algo_name == 'pca_annular_corr':
            indices_done = algo_dict['epoch_indices']
        else:
            indices_done = indices_cube_adi
        
        indices_epochs[N*2:(N*2)+2] = indices_done
        if algo_name == 'pca_annular' or algo_name == 'pca_annular_corr':
            _, res_cube_no_fc[:, indices_done[0]:indices_done[1], :, :], _ = algo(
                        cube=cube_adi, angle_list=this_angle_list, fwhm=fwhm_med,
                        verbose=verbose, full_output = True, **algo_dict)
        elif algo_name == 'pca_annular_multi_epoch' or algo_name == 'pca_annular_corr_multi_epoch':
            _, res_cube_no_fc[:, indices_done[0]:indices_done[1], :, :] = algo(
                        cube=cube, angle_list=angle_list, fwhm=fwhm_med,
                        verbose=verbose, full_output = True, **algo_dict)
        else:
            raise ValueError("Algorithm not supported")
    

    frames_basis_no_fc = np.median(res_cube_no_fc, axis = 1)

    noise_avg = np.array([noise_dist(frames_basis_no_fc[n, :, :], rad_dist, fwhm_med, wedge, 
                        False, debug) for n in range(0, nnpcs)])
    
    noise = noise_avg[:, :, 0]
    mean_res = noise_avg[:, :, 1]
    
    
    #minimizing noise in one half of annulus? look at imapct on source on other side
    
    #add option to kee pSNR above a threshold in contrast optimization
    
    #add approximatin parameters in optimization of contrast
    
    if source_xy is not None:
        
        x_source = source_xy[0]
        y_source = source_xy[1]
        snr_flux_basis = np.array([get_snr(frames_basis_no_fc[i], 
                    y_source, x_source, fwhm, fmerit) for i in range(0, nnpcs)])
        Optimal_comp_basis = ncomp[np.argmax(snr_flux_basis[:, 0])]
        
        Optimal_comp = np.full((nbr_epochs), Optimal_comp_basis)
        
            
        snr_progress = []
        flux_progress = []
        snr_progress.append(np.max(snr_flux_basis[:, 0]))
        flux_progress.append(snr_flux_basis[np.argmax(snr_flux_basis[:, 0]),1])
        
        #Optimize: when more than half values in Imporvements the same,
        #start simply selecting the first one that's better as an iteration
        
        #or a full otimization that always chooses the first better encountered
        #as the iteration
        
        #or step method, group epochs by groups, check all in the group, select the best
        #go to next group, do the same.. Repeat, each change = one iteration
        I = 0
        while I < iterations:
        #for I in range(iterations):
            Improvements = np.zeros((nbr_epochs,3))
            previous_N = -1
            no_found = 0
            for N in range(nbr_epochs):
                if N == previous_N:
                    continue
                
                snr_flux_test = np.zeros((nnpcs,2))
                for i, n in enumerate(ncomp):
                    res_cube_a = np.zeros((NbrImages, SizeImage, SizeImage))
                    
                    for Nbis in range(nbr_epochs):
                        #optimize: only change section of the epoch N in res_cube_a
                        #reset when Opt_comp has changed
                        if Nbis == N:
                            this_comp = i
                        else:
                            this_comp = int(np.where(ncomp == Optimal_comp[Nbis])[0])
                        res_cube_a[indices_epochs[Nbis*2]:
                            indices_epochs[(Nbis*2)+1],:,:] = res_cube_no_fc[
                            this_comp,indices_epochs[Nbis*2]:indices_epochs[(Nbis*2)+1],:,:]

                    this_frame = np.median(res_cube_a, axis = 0)
                    
                    snr_flux_test[i] = get_snr(this_frame, 
                                            y_source, x_source, fwhm, fmerit)
                
                if flux_increase:
                    sorted_ind = np.argsort(snr_flux_test[:,0])[::-1]
                    for ind in sorted_ind:
                        if snr_flux_test[ind,0] <= snr_progress[-1]:
                            break
                        if snr_flux_test[ind,1] <= flux_progress[-1]:
                            continue
                        Improvements[N,0:2] = snr_flux_test[ind,:]
                        Improvements[N,2] = ncomp[ind]
                        break
                else:
                    BestInd = np.argmax(snr_flux_test[:, 0])
                    Improvements[N,0:2] = snr_flux_test[BestInd,:]
                    Improvements[N,2] = ncomp[BestInd]
                
                if approximation == 3:
                    if Improvements[N,0] > snr_progress[-1]:
                        if flux_increase:
                            #sorted_ind = np.argsort(snr_flux_test[:,0])[::-1]
                            found = False
                            for ind in sorted_ind:
                                if snr_flux_test[ind,0] <= snr_progress[-1]:
                                    break
                                if snr_flux_test[ind,1] <= flux_progress[-1]:
                                    continue
                                found = True
                                I += 1
                                snr_progress.append(snr_flux_test[ind,0])
                                flux_progress.append(snr_flux_test[ind,1])
                                Optimal_comp[N] = ncomp[ind]
                                previous_N = N
                                if verbose:
                                    print(snr_progress[-1])
                                break
                            if found == False:
                                no_found += 1
                        else:
                            I += 1
                            snr_progress.append(Improvements[N,0])
                            Optimal_comp[N] = Improvements[N,2]
                            previous_N = N
                            if verbose:
                                print(snr_progress[-1])
                    else:
                        no_found += 1
                        
            if no_found == nbr_epochs:
                break
            if approximation == 0 or approximation == 1:
                #print(Improvements)
                Best_snr = np.max(Improvements[:,0])
                if Best_snr <= snr_progress[-1]:
                    print('Done after {}'.format(I))
                    break
                
                Best_N = np.argmax(Improvements[:, 0])
                
                if flux_increase:
                    if Improvements[Best_N,1]<=flux_progress[-1]:
                        print('Done after {}'.format(I))
                        break
                
                previous_N = Best_N
                Optimal_comp[Best_N] = Improvements[Best_N, 2]
                snr_progress.append(Best_snr)
                flux_progress.append(Improvements[Best_N,1])
                if verbose:
                    print(snr_progress[-1])
                
                if approximation == 1:
                    for j in range(nbr_epochs):
                        counter = 0
                        for  k in range(nbr_epochs):
                            if k == j:
                                continue
                            if Improvements[j,0] == Improvements[k,0]:
                                counter += 1
                            if counter > nbr_epochs/switch:
                                print("SWTICHED TO APPROXIMATION = 3")
                                approximation = 3
                                break
                        if approximation == 3:
                            break
            I+=1
        
        Best_frame_basis = frames_basis_no_fc[np.argmax(snr_flux_basis[:, 0])]
        for Nbis in range(nbr_epochs):
            this_comp = int(np.where(ncomp == Optimal_comp[Nbis])[0])
            res_cube_a[indices_epochs[Nbis*2]:
                indices_epochs[(Nbis*2)+1],:,:] = res_cube_no_fc[
                this_comp,indices_epochs[Nbis*2]:indices_epochs[(Nbis*2)+1],:,:]
        Best_frame = np.median(res_cube_a, axis = 0)
        
        return (Optimal_comp, Optimal_comp_basis, snr_flux_basis, snr_progress,
                flux_progress, Best_frame_basis, Best_frame)
    
    
    # We crop the PSF and check if PSF has been normalized (so that flux in
    # 1*FWHM aperture = 1) and fix if needed
    new_psf_size = int(round(3 * fwhm_med))
    if new_psf_size % 2 == 0:
        new_psf_size += 1
    
    n, y, x = cube.shape
    psf_template = normalize_psf(
        psf_template,
        fwhm=fwhm,
        verbose=verbose,
        size=min(new_psf_size, psf_template.shape[1]),
    )
    
    
    # Initialize the fake companions
    angle_branch = angular_range / nbranch
    
    Throughput = np.zeros((nnpcs, nbr_dist, nbranch))
    
    fc_map = np.zeros((y, x))
    cy, cx = frame_center(cube[0])
    parangles = angle_list

    # each branch is computed separately
    if matrix_adi_ref is not None:
        copy_ref = np.copy(algo_dict['cube_ref'])
        
    snr_basis = np.zeros((nnpcs, nbr_dist, nbranch))
    
    loc = np.zeros((nbr_dist, nbranch, 2))
    thruput_basis = np.zeros((nnpcs, nbr_dist, nbranch))
    recovered_flux_basis = np.zeros((nnpcs, nbr_dist, nbranch))
    all_injected_flux = np.zeros((nbr_dist, nbranch))
    for br in range(nbranch):
        
        if matrix_adi_ref is not None:
            algo_dict['cube_ref'] = np.copy(copy_ref)
        
        # each pattern is computed separately. For each one the companions
        # are separated by "fc_rad_sep * fwhm", interleaving the injections
        fc_map = np.ones_like(cube[0]) * 1e-6
        fcy = 0
        fcx = 0
        
        if flux is None:
            flux = fc_snr * np.array([np.percentile(noise_avg[:, d, 0], 20) for d in range(0,nbr_dist)])
        
        print(flux)
        
        if matrix_adi_ref is None:
            cube_fc = cube.copy()
        else:
            cube_fc = cube.copy()
            cube_adi_fc = np.copy(algo_dict['cube_ref'][NAdiRef:NRefT, :, :])
            cube_fc = np.vstack((cube_fc, cube_adi_fc))
            parangles = np.concatenate((angle_list, angle_adi_ref))
        
        for d in range(0, nbr_dist):
            cube_fc = cube_inject_companions(
                cube_fc,
                psf_template,
                parangles,
                flux[d],
                rad_dists=rad_dist[d],
                theta=br * angle_branch + theta,
                nproc=nproc,
                imlib=imlib,
                interpolation=interpolation,
                verbose=False,
                )
        
        if matrix_adi_ref is not None:
            algo_dict['cube_ref'][NAdiRef:NRefT, :, :] = cube_fc[n:, :, :]
            cube_fc = cube_fc[0:n, :, :]
            
        
        for d in range(0, nbr_dist):
            y = cy + rad_dist[d] * \
                np.sin(np.deg2rad(br * angle_branch + theta))
            x = cx + rad_dist[d] * \
                np.cos(np.deg2rad(br * angle_branch + theta))
            fc_map = frame_inject_companion(
                fc_map, psf_template, y, x, flux[d], imlib, interpolation
            )
            fcy = y
            fcx = x
            loc[d, br ,:] = np.array([fcy, fcx])

        if verbose:
            msg2 = "Fake companions injected in branch {} "
            print(msg2.format(br + 1))
            timing(start_time)
    
        for N in range(nbr_epochs):
            algo_dict = algo_dict_copy.copy()
            indices_cube_adi = (cube_delimiter[N+R*N], cube_delimiter[N+R*N+1])
            cube_adi = cube[indices_cube_adi[0]:indices_cube_adi[1]]
            this_angle_list = angle_list[cube_delimiter[N+R*N]:cube_delimiter[N+R*N+1]]
            if cube_ref_delimiter is not None:
                cube_ref_delimiter = np.array(cube_ref_delimiter)
                Rr = int(0)
                if cube_ref_delimiter.shape[0] == nbr_epochs*2:
                    Rr = int(1)
                indices_cube_rdi = (cube_ref_delimiter[N+Rr*N],cube_ref_delimiter[N+Rr*N+1])
                algo_dict['cube_ref'] = algo_dict['cube_ref'][indices_cube_rdi[0]:
                                                 indices_cube_rdi[1],:,:]
                    
            if algo.__name__ == 'pca_annular_corr':
                if epoch_indices is not None:
                    epoch_indices = np.array(epoch_indices)
                    Re = int(0)
                    if epoch_indices.shape[0] == nbr_epochs*2:
                        Re = int(1)
                    algo_dict['epoch_indices'] = epoch_indices[N+Re*N:N+Re*N+2]
                else:
                    algo_dict['epoch_indices'] = (indices_cube_adi[0],indices_cube_adi[1])
                
            if 'delta_rot' in algo_dict.keys():
                if isinstance(algo_dict['delta_rot'], list):
                    algo_dict['delta_rot'] = np.array(algo_dict['delta_rot'])
                if isinstance(algo_dict['delta_rot'], np.ndarray):
                    if algo_dict['delta_rot'].shape[0] != nbr_epochs:
                        raise ValueError('delta_rot has wrong length')
                    algo_dict['delta_rot'] = algo_dict['delta_rot'][N]
            
            if algo_name == 'pca_annular_corr':
                indices_done = algo_dict['epoch_indices']
            else:
                indices_done = indices_cube_adi
            if algo_name == 'pca_annular' or algo_name == 'pca_annular_corr':
                _, res_cube_fc[:,br,indices_done[0]:indices_done[1],:,:], _ = algo(
                    cube=cube_fc, angle_list=angle_list, fwhm=fwhm_med, 
                    verbose=verbose, full_output = True, **algo_dict)
            elif algo_name == 'pca_annular_multi_epoch' or algo_name == 'pca_annular_corr_multi_epoch':
                _, res_cube_fc[:,br,indices_done[0]:indices_done[1],:,:] = algo(
                    cube=cube_fc, angle_list=angle_list, fwhm=fwhm_med, 
                    verbose=verbose, full_output = True, **algo_dict)
        
        frames_basis_fc[:,br,:,:] = np.median(res_cube_fc[:,br,:,:,:], axis = 1)

        injected_flux = apertureOne_flux(fc_map, loc[:,br,0], loc[:,br,1], fwhm_med)
        all_injected_flux[:,br] = injected_flux
        recovered_flux_avg = np.array([apertureOne_flux(
            (frames_basis_fc[n, br, :, :] - frames_basis_no_fc[n, :, :]), loc[:,br,0], loc[:,br,1], fwhm_med
        ) for n in range(0, nnpcs)])
        recovered_flux_basis[:,:,br] = recovered_flux_avg
        for d in range(0, nbr_dist):
            thruput_basis[:,d,br] = recovered_flux_avg[:,d]/injected_flux[d]
        thruput_basis[np.where(thruput_basis < 0)] = 0
        
        for d in range(0, nbr_dist):
            snr_basis[:, d, br] = [get_snr(frames_basis_fc[n,br,:,:], 
                loc[d,br,0], loc[d,br,1], fwhm, fmerit)[0] for n in range(0,nnpcs)]
        

    snr_basis = np.median(snr_basis, axis = 2)
    thruput_basis = np.nanmean(thruput_basis[:,:,:], axis = 2)
    thru_cont_basis = np.zeros((nnpcs, nbr_dist, 3))
    thru_cont_basis[:,:,0] = thruput_basis
    if isinstance(starphot, float) or isinstance(starphot, int):
        thru_cont_basis[:,:,1] = (
            (sigma * noise_avg[:,:,0]) / thru_cont_basis[:,:,0]
        ) / starphot
    else:
        thru_cont_basis[:,:,1] = (
            (sigma * noise_avg[:,:,0]) / thru_cont_basis[:,:,0]
        ) / np.median(starphot)
    if 'multi_epoch' not in algo_name:
        thru_cont_basis[:,:,2] = ncomp.reshape(ncomp.shape[0],1)
    
    print(thru_cont_basis)
    print(snr_basis)
        
    Optimal_comp_basis = []
    if through_thresh != 'auto':
        for d in range(nbr_dist):
            Optimal_comp_basis.append(ncomp[np.argmin(
                thru_cont_basis[thru_cont_basis[:,d,0]>through_thresh, d, 1])])
    else:
        for d in range(nbr_dist):
            Optimal_comp_basis.append(ncomp[np.argmin(thru_cont_basis[:, d, 1])])
    Optimal_comp_basis = np.array(Optimal_comp_basis)
    
    if through_thresh == 'auto':
        through_thresh = [thru_cont_basis[np.argmin(thru_cont_basis[:, d, 1]),d,0] 
                                                          for d in range(nbr_dist)]
    elif np.isscalar(through_thresh):
        through_thresh = [thru_cont_basis[np.argmin(thru_cont_basis[
            thru_cont_basis[:,d,0]>through_thresh, d, 1]),d,0] for d in range(nbr_dist)]
    
    
    
        
    
    Optimal_comp = np.zeros((nbr_epochs, nbr_dist), dtype = int)
    for i in range(nbr_epochs):
        Optimal_comp[i,:] = Optimal_comp_basis.copy()
        
    Contrast_progress = []
    for d in range(nbr_dist):
        Contrast_progress.append([np.min(thru_cont_basis[:, d, 1])])
        
    Done = []
    I = 0
    while I < iterations:
        Improvements = np.zeros((nbr_epochs,nbr_dist,3))
        
        no_found = np.zeros(nbr_epochs)
        for N in range(nbr_epochs):
            
            for d in range(nbr_dist):
                if d in Done:
                    continue
                
                contrasts = np.zeros((nnpcs,2))
                for i, n in enumerate(ncomp):
                    res_cube_a = np.zeros((NbrImages, SizeImage, SizeImage))
                    res_cube_fc_a = np.zeros((nbranch, NbrImages, SizeImage, SizeImage))
                    
                    for Nbis in range(nbr_epochs):
                        if Nbis == N:
                            this_comp = i
                        else:
                            this_comp = int(np.where(ncomp == Optimal_comp[Nbis,d])[0])
                        res_cube_a[indices_epochs[Nbis*2]:
                            indices_epochs[(Nbis*2)+1],:,:] = res_cube_no_fc[
                            this_comp,indices_epochs[Nbis*2]:indices_epochs[(Nbis*2)+1],:,:]
                        res_cube_fc_a[indices_epochs[Nbis*2]:
                            indices_epochs[(Nbis*2)+1],:,:] = res_cube_fc[
                            this_comp,indices_epochs[Nbis*2]:indices_epochs[(Nbis*2)+1],:,:]
                    
                    this_frame = np.median(res_cube_a, axis = 0)
                    this_frames_fc = np.median(res_cube_fc_a, axis = 1)
                    
                    this_noise, this_mean = noise_dist(this_frame, rad_dist[d], fwhm_med, wedge, 
                                        False, debug)[0]
                    #print(this_noise, this_mean)
                
                    this_thruput = np.zeros(nbranch)
                    this_flux = np.array([apertureOne_flux(
                        (this_frames_fc[br, :, :] - this_frame), loc[d,br,0], loc[d,br,1], fwhm_med
                    ) for br in range(nbranch)])
                    
                    for br in range(nbranch):
                        this_thruput[br] = this_flux[br]/all_injected_flux[d, br]
                        
                    this_thruput[np.where(this_thruput < 0)] = 0
                    this_thruput[np.where(this_thruput > 1)] = 0
                    
                    #if np.sum(this_thruput > through_thresh[d]) < nbranch:
                    #    this_thruput = 0
                    #print(this_thruput)
                    if len(np.where(this_thruput > 0)[0]) == 0:
                        contrasts[i,0] = 0
                        contrasts[i,1] = 1
                        continue
                    else:
                        this_avg_thruput = np.nanmean(this_thruput)
                        
                    if this_avg_thruput < through_thresh[d]:
                        contrasts[i,0] = 0
                        contrasts[i,1] = 1
                        continue
                    
                    if isinstance(starphot, float) or isinstance(starphot, int):
                        this_contrast = (
                            (sigma * this_noise) / this_avg_thruput
                        ) / starphot
                    else:
                        this_contrast = (
                            (sigma * this_noise) / this_avg_thruput
                        ) / np.median(starphot)
                    
                    
                    contrasts[i,0] = this_avg_thruput
                    contrasts[i,1] = this_contrast
                #print(contrasts)
                #if np.sum(contrasts == 1) == nnpcs:
                #    continue
                #Optimal_comp[N,d] = ncomp[np.argmin(contrasts)]
                #if np.min(contrasts) < Contrast_progress[int(d)][-1]:
                #    Contrast_progress[d].append(np.min(contrasts))
                
                if np.sum(contrasts[:,1] == 1) == nnpcs:
                    Improvements[N,d,1] = 1
                    Improvements[N,d,0] = Optimal_comp[N,d]
                else:
                    #print(np.min(contrasts[:,1]))
                    BestCompInd = np.argmin(contrasts[:,1])
                    Improvements[N,d,1] = np.min(contrasts[:,1])
                    Improvements[N,d,0] = ncomp[BestCompInd]
                    Improvements[N,d,2] = contrasts[BestCompInd,0]
        
                if approximation == 3:
                    if Improvements[N,d,1] < Contrast_progress[d][-1]:
                        Optimal_comp[N,d] = Improvements[N,d,0]
                        Contrast_progress[d].append(Improvements[N,d,1])
                        through_thresh[d] = Improvements[N,d,2]
                        I+=1
                    else:
                        no_found[d] +=1
        
        #print(Improvements)
        for d in range(nbr_dist):
            if d in Done:
                continue
            
            if approximation == 3:
                if no_found[d] == nbr_epochs:
                    Done.append(d)
                    continue
            
            if approximation == 0:
                Best_C = np.min(Improvements[:,d,1])
                if Best_C >= Contrast_progress[int(d)][-1]:
                    Done.append(d)
                    continue
                Best_N = np.argmin(Improvements[:,d,1])
                Optimal_comp[Best_N,d] = Improvements[Best_N,d,0]
                Contrast_progress[d].append(Best_C)
                through_thresh[d] = Improvements[Best_N,d,2]
                #print(through_thresh)
                I+=1
                
        #print(Optimal_comp)
        if len(Done) == nbr_dist:
            print('Done after {}'.format(I))
            break
    
    return Optimal_comp, Optimal_comp_basis, Contrast_progress



def contrast_multi_epoch_walk2(
    cube,
    angle_list,
    psf_template,
    fwhm,
    distance,
    pxscale,
    starphot,
    algo,
    nbr_cubes,
    step_walk,
    iterations=5,
    through_thresh=0.1,
    through_up=True,
    snr_thresh=0,
    sigma=5,
    nbranch=1,
    theta=0,
    inner_rad=1,
    fc_rad_sep=3,
    approximation = 0,
    switch = 3,
    flux_increase = False,
    noise_sep=1,
    wedge=(0, 360),
    fc_snr=50,
    snr_target=10,
    per_values = [70, 10],
    flux = None,
    cube_delimiter=None,
    cube_ref_delimiter=None,
    epoch_indices=None,
    student=True,
    transmission=None,
    dpi=vip_figdpi,
    debug=False,
    verbose=True,
    full_output=False,
    algo_class=None,
    matrix_adi_ref=None,
    angle_adi_ref=None,
    source_xy = None,
    exclude_negative_lobes=False,
    fmerit = 'mean',
    **algo_dict,
):
    """
    -step: contains the number of image in each epoch. If an integer, all epoch
    will have the same number of images
    
    -nbr_cubes: the number of subdivisions in the processing of the cube
    
    -through_thresh:Threshold put on thhe throughput for it to be considered valid
    if through_tresh is 'auto', the threshold is chosen as the throughput of the
    best contrast found in the basis of ncomp tested
    
    -through_up: if True, any contrast value that improves upon the last must also
    increase (or maintain) the throughput to be considered valid
    
    -per_values: percentile values for the fluxes of the fake companions that 
    are injected in the data cube. It then allows for interpolation of the
    throughput for any injected flux
    
    -approximation: Determines the level of approximation to optimize the contrast
    or the snr depending on the mode.
    approximation == 0 only recommended for small datasets, or a small number of 
    epochs. 
    
    -flux: Fix the flux injected for the fake companions. Takes precedence over
    per_values if it is not None.
    
    -iterations: Maximum number of iterations the algorithm will go through
    
    -source_xy : if it is not None, then the function is in snr optimization
    mode. It will try to find the component that increases the snr of the source
    given.
    
    -flux_increase: if source_xy is not None and flux_increase is True, any 
    change of components that increases the snr must also increase the flux of 
    the source.
    """
    
    from ..metrics import snr
    def get_snr(frame, y, x, fwhm, fmerit):
        """
        """
        if fmerit == 'max':
            yy, xx = disk((y, x), fwhm / 2.)
            res = [snr(frame, (x_, y_), fwhm, plot=False, verbose=False,
                       exclude_negative_lobes=exclude_negative_lobes,
                       full_output=True)
                   for y_, x_ in zip(yy, xx)]
            snr_pixels = np.array(res, dtype=object)[:, -1]
            fluxes = np.array(res, dtype=object)[:, 2]
            argm = np.argmax(snr_pixels)
            # integrated fluxes for the max snr
            return np.max(snr_pixels), fluxes[argm]

        elif fmerit == 'px':
            res = snr(frame, (x, y), fwhm, plot=False, verbose=False,
                      exclude_negative_lobes=exclude_negative_lobes,
                      full_output=True)
            snrpx = res[-1]
            fluxpx = np.array(res, dtype=object)[2]
            # integrated fluxes for the given px
            return snrpx, fluxpx

        elif fmerit == 'mean':
            yy, xx = disk((y, x), fwhm / 2.)
            res = [snr(frame, (x_, y_), fwhm, plot=False, verbose=False,
                       exclude_negative_lobes=exclude_negative_lobes,
                       full_output=True) for y_, x_
                   in zip(yy, xx)]
            snr_pixels = np.array(res, dtype=object)[:, -1]
            fluxes = np.array(res, dtype=object)[:, 2]
            # mean of the integrated fluxes (shifting the aperture)
            return np.mean(snr_pixels), np.mean(fluxes)
        
    def interpol(x, xp, yp):
        xp = np.array(xp)
        yp = np.array(yp)
        indices = np.argsort(xp)
        xp = xp[indices]
        yp = yp[indices]
        if np.isscalar(x):
            x = np.array([x])
        else:
            x = np.array(x)
        for value in x:
            if value < xp[0]:
                slope = (yp[1]-yp[0])/(xp[1]-xp[0])
                return yp[0]+(value-xp[0])*slope
            elif value > xp[-1]:
                slope = (yp[-1]-yp[-2])/(xp[-1]-xp[-2])
                return yp[-1]+(value-xp[-1])*slope
            else:
                for i, v in enumerate(xp):
                    if value > v:
                        continue
                    slope = (yp[i]-yp[i-1])/(xp[i]-xp[i-1])
                    return yp[i-1]+(value-xp[i-1])*slope
        
    #-----------------------------------------------------------------

    if cube.ndim != 3 and cube.ndim != 4:
        raise TypeError("The input array is not a 3d or 4d cube")
    if cube.ndim == 3 and (cube.shape[0] != angle_list.shape[0]):
        raise TypeError("Input parallactic angles vector has wrong length")
    if cube.ndim == 4 and (cube.shape[1] != angle_list.shape[0]):
        raise TypeError("Input parallactic angles vector has wrong length")
    if cube.ndim == 3 and psf_template.ndim != 2:
        raise TypeError("Template PSF is not a frame (for ADI case)")
    if cube.ndim == 4 and psf_template.ndim != 3:
        raise TypeError("Template PSF is not a cube (for ADI+IFS case)")
    if transmission is not None:
        if len(transmission) != 2 and len(transmission) != cube.shape[0] + 1:
            msg = "Wrong shape for transmission should be 2xn_rad or (nch+1) "
            msg += "x n_rad, instead of {}".format(transmission.shape)
            raise TypeError(msg)

    if isinstance(fwhm, (np.ndarray, list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if verbose:
        start_time = time_ini()
        if isinstance(starphot, float) or isinstance(starphot, int):
            msg0 = "ALGO : {}, FWHM = {}, # BRANCHES = {}, SIGMA = {},"
            msg0 += " STARPHOT = {}"
            print(msg0.format(algo.__name__, fwhm_med, nbranch, sigma, starphot))
        else:
            msg0 = "ALGO : {}, FWHM = {}, # BRANCHES = {}, SIGMA = {}"
            print(msg0.format(algo.__name__, fwhm_med, nbranch, sigma))
            
    if cube.ndim != 3 and cube.ndim != 4:
        raise TypeError("The input array is not a 3d or 4d cube")

    else:
        if psf_template.shape[1] % 2 == 0:
            raise ValueError("Only odd-sized PSF is accepted")
        if not hasattr(algo, "__call__"):
            raise TypeError("Parameter `algo` must be a callable function")
        if not isinstance(inner_rad, int):
            raise TypeError("inner_rad must be an integer")
        angular_range = wedge[1] - wedge[0]
        if nbranch > 1 and angular_range < 360:
            msg = "Only a single branch is allowed when working on a wedge"
            raise RuntimeError(msg)
        
    nproc = algo_dict.get("nproc", 1)
    imlib = algo_dict.get("imlib", "vip-fft")
    interpolation = algo_dict.get("interpolation", "lanczos4")
    scaling = algo_dict.get("scaling", None)
    ncomp = algo_dict.get("ncomp")
    
    nnpcs = len(ncomp)
    
    if np.isscalar(ncomp):
        ncomp = np.array([ncomp])
    elif isinstance(ncomp, list):
        ncomp = np.array(ncomp)
            
    algo_name = algo.__name__
    idx = algo.__module__.index('.', algo.__module__.index('.') + 1)
    mod = algo.__module__[:idx]
    tmp = __import__(mod, fromlist=[algo_name.upper()+'_Params'])    
    algo_params = getattr(tmp, algo_name.upper()+'_Params')
    
    algo_supported = ['pca_annular', 'pca_annular_corr', 
                      'pca_annular_multi_epoch', 'pca_annular_corr_multi_epoch']
    if algo_name not in algo_supported:
        raise ValueError("Algorithm is not supported")
    
    if isinstance(fwhm, (np.ndarray, list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if verbose:
        start_time = time_ini()
        
    
    if matrix_adi_ref is not None:
        if 'cube_ref' in algo_dict.keys() and algo_dict['cube_ref'] is not None:
            NAdiRef = algo_dict['cube_ref'].shape[0]
            algo_dict['cube_ref'] = np.vstack((algo_dict['cube_ref'], matrix_adi_ref))
        else:
            NAdiRef = 0
            algo_dict['cube_ref'] = matrix_adi_ref
        NRefT = algo_dict['cube_ref'].shape[0]
    
    if 'annular' in algo_name:
        if distance == 'auto':
            radius_int = algo_dict['radius_int']
            asize = algo_dict['asize']
            y = cube.shape[2]
            n_annuli = int((y / 2 - radius_int) / asize)
            distance = np.array([radius_int+(asize/2) + i*asize for i in range(0, n_annuli)])/fwhm_med
        elif isinstance(distance, float):
            distance = np.array([distance])
        elif isinstance(distance, list):
            distance = np.array(distance)
        else:
            raise ValueError("distance parameter must be a float, a list or equal to 'auto'")
            
    SizeImage = int(cube[0].shape[1])
    NbrImages = int(cube.shape[0])
    
    frames_basis_fc = np.zeros((nnpcs, nbranch, SizeImage, SizeImage), dtype = float)
    frames_basis_no_fc = np.zeros((nnpcs, SizeImage, SizeImage), dtype = float)
    res_cube_fc = np.zeros((nnpcs, nbranch, NbrImages, SizeImage, SizeImage), dtype = float)
    res_cube_no_fc = np.zeros((nnpcs, NbrImages, SizeImage, SizeImage), dtype = float)    
    
    rad_dist = distance * fwhm_med
    nbr_dist = distance.shape[0]
    
    results = []
    algo_dict_copy = algo_dict.copy()
    
    if np.isscalar(step_walk):
        nbr_epochs = int(NbrImages/step_walk)
        step_walk = [step_walk]*nbr_epochs
    else:
        nbr_epochs = len(step_walk)
    
    if isinstance(cube_delimiter, list):
        cube_delimiter = np.array(cube_delimiter)
    if cube_delimiter.shape[0] == nbr_cubes*2:
        R = int(1)
    else:
        R = int(0)

    indices_epochs = np.zeros(nbr_epochs*2, dtype = int)
    
    previous_e = 0
    for n in range(nbr_epochs):
        indices_epochs[n*2:(n*2)+2] = (previous_e, previous_e + step_walk[n])
        previous_e += step_walk[n]
    
    epoch_saved = np.copy(epoch_indices)
        
    for N in range(nbr_cubes):
        algo_dict = algo_dict_copy.copy()
        
        if len(epoch_indices) == 2 and nbr_cubes > 1:
            algo_dict['epoch_indices'] = epoch_indices
            _, res_cube_no_fc, _ = algo(
                cube=cube, angle_list=angle_list, fwhm=fwhm_med,
                verbose=verbose, full_output = True, **algo_dict)
            step = step_walk[N]
            epoch_indices = [0]
            for n in range(nbr_epochs):
                epoch_indices.append(epoch_indices[-1]+step_walk[n])
            indices_epochs = [epoch_indices[int(j/2) + j%2] for j in range(0, nbr_epochs*2)]
            break
        
        indices_cube_adi = (cube_delimiter[N+R*N], cube_delimiter[N+R*N+1])
        cube_adi = cube[indices_cube_adi[0]:indices_cube_adi[1]]
        this_angle_list = angle_list[indices_cube_adi[0]:indices_cube_adi[1]]
        if cube_ref_delimiter is not None:
            cube_ref_delimiter = np.array(cube_ref_delimiter)
            Rr = int(0)
            if cube_ref_delimiter.shape[0] == nbr_cubes*2:
                Rr = int(1)
            indices_cube_rdi = (cube_ref_delimiter[N+Rr*N],cube_ref_delimiter[N+Rr*N+1])
            algo_dict['cube_ref'] = algo_dict['cube_ref'][indices_cube_rdi[0]:
                                             indices_cube_rdi[1],:,:]
                
        if algo.__name__ == 'pca_annular_corr':
            if epoch_indices is not None:
                epoch_indices = np.array(epoch_indices)
                Re = int(0)
                if epoch_indices.shape[0] == nbr_cubes*2:
                    Re = int(1)
                algo_dict['epoch_indices'] = epoch_indices[N+Re*N:N+Re*N+2]
            else:
                algo_dict['epoch_indices'] = (indices_cube_adi[0],indices_cube_adi[1])
            
        if 'delta_rot' in algo_dict.keys():
            if isinstance(algo_dict['delta_rot'], list):
                algo_dict['delta_rot'] = np.array(algo_dict['delta_rot'])
            if isinstance(algo_dict['delta_rot'], np.ndarray):
                if algo_dict['delta_rot'].shape[0] != nbr_cubes:
                    raise ValueError('delta_rot has wrong length')
                algo_dict['delta_rot'] = algo_dict['delta_rot'][N]
        
        
        if algo_name == 'pca_annular_corr':
            indices_done = algo_dict['epoch_indices']
        else:
            indices_done = indices_cube_adi
        
        if algo_name == 'pca_annular' or algo_name == 'pca_annular_corr':
            _, res_cube_no_fc[:, indices_done[0]:indices_done[1], :, :], _ = algo(
                        cube=cube_adi, angle_list=this_angle_list, fwhm=fwhm_med,
                        verbose=verbose, full_output = True, **algo_dict)
        elif algo_name == 'pca_annular_multi_epoch' or algo_name == 'pca_annular_corr_multi_epoch':
            _, res_cube_no_fc[:, indices_done[0]:indices_done[1], :, :] = algo(
                        cube=cube, angle_list=angle_list, fwhm=fwhm_med,
                        verbose=verbose, full_output = True, **algo_dict)
        else:
            raise ValueError("Algorithm not supported")
          
    frames_per_epoch = [(indices_epochs[i+1]-indices_epochs[i])/NbrImages
                                    for i in range(0, len(indices_epochs), 2)]
    #print(frames_per_epoch)

    frames_basis_no_fc = np.median(res_cube_no_fc, axis = 1)

    noise_avg = np.array([noise_dist(frames_basis_no_fc[n, :, :], rad_dist, fwhm_med, wedge, 
                        False, debug) for n in range(0, nnpcs)])
    
    noise = noise_avg[:, :, 0]
    mean_res = noise_avg[:, :, 1]
    
    
    #minimizing noise in one half of annulus? look at imapct on source on other side
    
    #add option to kee pSNR above a threshold in contrast optimization
    
    #add approximatin parameters in optimization of contrast
    
    if source_xy is not None:
        
        x_source = source_xy[0]
        y_source = source_xy[1]
        snr_flux_basis = np.array([get_snr(frames_basis_no_fc[i], 
                    y_source, x_source, fwhm, fmerit) for i in range(0, nnpcs)])
        Optimal_comp_basis = ncomp[np.argmax(snr_flux_basis[:, 0])]
        
        Optimal_comp = np.full((nbr_epochs), Optimal_comp_basis)
        
            
        snr_progress = []
        flux_progress = []
        snr_progress.append(np.max(snr_flux_basis[:, 0]))
        flux_progress.append(snr_flux_basis[np.argmax(snr_flux_basis[:, 0]),1])
        
        #Optimize: when more than half values in Imporvements the same,
        #start simply selecting the first one that's better as an iteration
        
        #or a full otimization that always chooses the first better encountered
        #as the iteration
        
        #or step method, group epochs by groups, check all in the group, select the best
        #go to next group, do the same.. Repeat, each change = one iteration
        I = 0
        while I < iterations:
        #for I in range(iterations):
            Improvements = np.zeros((nbr_epochs,3))
            previous_N = -1
            no_found = 0
            for N in range(nbr_epochs):
                if N == previous_N:
                    continue
                
                snr_flux_test = np.zeros((nnpcs,2))
                for i, n in enumerate(ncomp):
                    res_cube_a = np.zeros((NbrImages, SizeImage, SizeImage))
                    
                    for Nbis in range(nbr_epochs):
                        #optimize: only change section of the epoch N in res_cube_a
                        #reset when Opt_comp has changed
                        if Nbis == N:
                            this_comp = i
                        else:
                            this_comp = int(np.where(ncomp == Optimal_comp[Nbis])[0])
                        res_cube_a[indices_epochs[Nbis*2]:
                            indices_epochs[(Nbis*2)+1],:,:] = res_cube_no_fc[
                            this_comp,indices_epochs[Nbis*2]:indices_epochs[(Nbis*2)+1],:,:]

                    this_frame = np.median(res_cube_a, axis = 0)
                    
                    snr_flux_test[i] = get_snr(this_frame, 
                                            y_source, x_source, fwhm, fmerit)
                
                if flux_increase:
                    sorted_ind = np.argsort(snr_flux_test[:,0])[::-1]
                    for ind in sorted_ind:
                        if snr_flux_test[ind,0] <= snr_progress[-1]:
                            break
                        if snr_flux_test[ind,1] <= flux_progress[-1]:
                            continue
                        Improvements[N,0:2] = snr_flux_test[ind,:]
                        Improvements[N,2] = ncomp[ind]
                        break
                else:
                    BestInd = np.argmax(snr_flux_test[:, 0])
                    Improvements[N,0:2] = snr_flux_test[BestInd,:]
                    Improvements[N,2] = ncomp[BestInd]
                
                if approximation == 3:
                    if Improvements[N,0] > snr_progress[-1]:
                        if flux_increase:
                            #sorted_ind = np.argsort(snr_flux_test[:,0])[::-1]
                            found = False
                            for ind in sorted_ind:
                                if snr_flux_test[ind,0] <= snr_progress[-1]:
                                    break
                                if snr_flux_test[ind,1] <= flux_progress[-1]:
                                    continue
                                found = True
                                I += 1
                                snr_progress.append(snr_flux_test[ind,0])
                                flux_progress.append(snr_flux_test[ind,1])
                                Optimal_comp[N] = ncomp[ind]
                                previous_N = N
                                if verbose:
                                    print(snr_progress[-1])
                                break
                            if found == False:
                                no_found += 1
                        else:
                            I += 1
                            snr_progress.append(Improvements[N,0])
                            Optimal_comp[N] = Improvements[N,2]
                            previous_N = N
                            if verbose:
                                print(snr_progress[-1])
                    else:
                        no_found += 1
                        
            if no_found == nbr_epochs:
                break
            if approximation == 0 or approximation == 1:
                #print(Improvements)
                Best_snr = np.max(Improvements[:,0])
                if Best_snr <= snr_progress[-1]:
                    print('Done after {}'.format(I))
                    break
                
                Best_N = np.argmax(Improvements[:, 0])
                
                if flux_increase:
                    if Improvements[Best_N,1]<=flux_progress[-1]:
                        print('Done after {}'.format(I))
                        break
                
                previous_N = Best_N
                Optimal_comp[Best_N] = Improvements[Best_N, 2]
                snr_progress.append(Best_snr)
                flux_progress.append(Improvements[Best_N,1])
                if verbose:
                    print(snr_progress[-1])
                
                if approximation == 1:
                    for j in range(nbr_epochs):
                        counter = 0
                        for  k in range(nbr_epochs):
                            if k == j:
                                continue
                            if Improvements[j,0] == Improvements[k,0]:
                                counter += 1
                            if counter > nbr_epochs/switch:
                                print("SWTICHED TO APPROXIMATION = 3")
                                approximation = 3
                                break
                        if approximation == 3:
                            break
            I+=1
        
        Best_frame_basis = frames_basis_no_fc[np.argmax(snr_flux_basis[:, 0])]
        for Nbis in range(nbr_epochs):
            this_comp = int(np.where(ncomp == Optimal_comp[Nbis])[0])
            res_cube_a[indices_epochs[Nbis*2]:
                indices_epochs[(Nbis*2)+1],:,:] = res_cube_no_fc[
                this_comp,indices_epochs[Nbis*2]:indices_epochs[(Nbis*2)+1],:,:]
        Best_frame = np.median(res_cube_a, axis = 0)
        
        return (Optimal_comp, Optimal_comp_basis, snr_flux_basis, snr_progress,
                flux_progress, Best_frame_basis, Best_frame)
    
    
    # We crop the PSF and check if PSF has been normalized (so that flux in
    # 1*FWHM aperture = 1) and fix if needed
    new_psf_size = int(round(3 * fwhm_med))
    if new_psf_size % 2 == 0:
        new_psf_size += 1
    
    n, y, x = cube.shape
    psf_template = normalize_psf(
        psf_template,
        fwhm=fwhm,
        verbose=verbose,
        size=min(new_psf_size, psf_template.shape[1]),
    )
    
    
    # Initialize the fake companions
    angle_branch = angular_range / nbranch
    
    Throughput = np.zeros((nnpcs, nbr_dist, nbranch))
    
    fc_map = np.zeros((y, x))
    cy, cx = frame_center(cube[0])
    parangles = angle_list

    # each branch is computed separately
    if matrix_adi_ref is not None:
        copy_ref = np.copy(algo_dict['cube_ref'])
        
    nbr_per = len(per_values)
    snr_basis_per = np.zeros((nnpcs, nbr_dist, nbranch, nbr_per))
    
    loc = np.zeros((nbr_dist, nbranch, 2))
    thruput_basis = np.zeros((nnpcs, nbr_dist, nbranch))
    recovered_flux_basis = np.zeros((nnpcs, nbr_dist, nbranch))
    all_injected_flux = np.zeros((nbr_dist, nbranch))
    
    if np.isscalar(fc_snr):
        fc_snr = np.array(fc_snr)
    
    thruput_per = np.zeros((nnpcs, nbr_dist, nbr_per))
    this_flux = flux
    all_fluxes = np.zeros((nbr_dist, nbr_per))
    for i, per in enumerate(per_values):
        
        for br in range(nbranch):
        
            if matrix_adi_ref is not None:
                algo_dict['cube_ref'] = np.copy(copy_ref)
        
            # each pattern is computed separately. For each one the companions
            # are separated by "fc_rad_sep * fwhm", interleaving the injections
            fc_map = np.ones_like(cube[0]) * 1e-6
            fcy = 0
            fcx = 0
        
            if this_flux is None:
                flux = np.array(fc_snr) * np.array([np.percentile(noise_avg[:, d, 0], per) for d in range(0,nbr_dist)])
                
            if br == 0:
                all_fluxes[:,i] = flux
            print(flux)
        
            if matrix_adi_ref is None:
                cube_fc = cube.copy()
            else:
                cube_fc = cube.copy()
                cube_adi_fc = np.copy(algo_dict['cube_ref'][NAdiRef:NRefT, :, :])
                cube_fc = np.vstack((cube_fc, cube_adi_fc))
                parangles = np.concatenate((angle_list, angle_adi_ref))
        
            for d in range(0, nbr_dist):
                cube_fc = cube_inject_companions(
                    cube_fc,
                    psf_template,
                    parangles,
                    flux[d],
                    rad_dists=rad_dist[d],
                    theta=br * angle_branch + theta,
                    nproc=nproc,
                    imlib=imlib,
                    interpolation=interpolation,
                    verbose=False,
                    )
        
            if matrix_adi_ref is not None:
                algo_dict['cube_ref'][NAdiRef:NRefT, :, :] = cube_fc[n:, :, :]
                cube_fc = cube_fc[0:n, :, :]
            
        
            for d in range(0, nbr_dist):
                y = cy + rad_dist[d] * \
                    np.sin(np.deg2rad(br * angle_branch + theta))
                x = cx + rad_dist[d] * \
                    np.cos(np.deg2rad(br * angle_branch + theta))
                fc_map = frame_inject_companion(
                    fc_map, psf_template, y, x, flux[d], imlib, interpolation
                )
                fcy = y
                fcx = x
                loc[d, br ,:] = np.array([fcy, fcx])

            if verbose:
                msg2 = "Fake companions injected in branch {} "
                print(msg2.format(br + 1))
                timing(start_time)
    
            for N in range(nbr_cubes):
                algo_dict = algo_dict_copy.copy()
                
                if len(epoch_saved) == 2:
                    algo_dict['epoch_indices'] = epoch_saved
                    _, res_cube_fc[:,br,:,:,:], _ = algo(
                        cube=cube_fc, angle_list=angle_list, fwhm=fwhm_med, 
                        verbose=verbose, full_output = True, **algo_dict)
                    break
                
                indices_cube_adi = (cube_delimiter[N+R*N], cube_delimiter[N+R*N+1])
                cube_adi = cube[indices_cube_adi[0]:indices_cube_adi[1]]
                this_angle_list = angle_list[cube_delimiter[N+R*N]:cube_delimiter[N+R*N+1]]
                if cube_ref_delimiter is not None:
                    cube_ref_delimiter = np.array(cube_ref_delimiter)
                    Rr = int(0)
                    if cube_ref_delimiter.shape[0] == nbr_cubes*2:
                        Rr = int(1)
                    indices_cube_rdi = (cube_ref_delimiter[N+Rr*N],cube_ref_delimiter[N+Rr*N+1])
                    algo_dict['cube_ref'] = algo_dict['cube_ref'][indices_cube_rdi[0]:
                                                 indices_cube_rdi[1],:,:]
                    
                if algo.__name__ == 'pca_annular_corr':
                    if epoch_indices is not None:
                        epoch_indices = np.array(epoch_indices)
                        Re = int(0)
                        if epoch_indices.shape[0] == nbr_cubes*2:
                            Re = int(1)
                        algo_dict['epoch_indices'] = epoch_indices[N+Re*N:N+Re*N+2]
                    else:
                        algo_dict['epoch_indices'] = (indices_cube_adi[0],indices_cube_adi[1])
                
                if 'delta_rot' in algo_dict.keys():
                    if isinstance(algo_dict['delta_rot'], list):
                        algo_dict['delta_rot'] = np.array(algo_dict['delta_rot'])
                    if isinstance(algo_dict['delta_rot'], np.ndarray):
                        if algo_dict['delta_rot'].shape[0] != nbr_cubes:
                            raise ValueError('delta_rot has wrong length')
                        algo_dict['delta_rot'] = algo_dict['delta_rot'][N]
            
                if algo_name == 'pca_annular_corr':
                    indices_done = algo_dict['epoch_indices']
                else:
                    indices_done = indices_cube_adi
                if algo_name == 'pca_annular' or algo_name == 'pca_annular_corr':
                    _, res_cube_fc[:,br,indices_done[0]:indices_done[1],:,:], _ = algo(
                        cube=cube_fc, angle_list=angle_list, fwhm=fwhm_med, 
                        verbose=verbose, full_output = True, **algo_dict)
                elif algo_name == 'pca_annular_multi_epoch' or algo_name == 'pca_annular_corr_multi_epoch':
                    _, res_cube_fc[:,br,indices_done[0]:indices_done[1],:,:] = algo(
                        cube=cube_fc, angle_list=angle_list, fwhm=fwhm_med, 
                        verbose=verbose, full_output = True, **algo_dict)
        
            frames_basis_fc[:,br,:,:] = np.median(res_cube_fc[:,br,:,:,:], axis = 1)

            injected_flux = apertureOne_flux(fc_map, loc[:,br,0], loc[:,br,1], fwhm_med)
            
            all_injected_flux[:,br] = injected_flux
            recovered_flux_avg = np.array([apertureOne_flux(
                (frames_basis_fc[n, br, :, :] - frames_basis_no_fc[n, :, :]), loc[:,br,0], loc[:,br,1], fwhm_med
            ) for n in range(0, nnpcs)])
            recovered_flux_basis[:,:,br] = recovered_flux_avg
            for d in range(0, nbr_dist):
                thruput_basis[:,d,br] = recovered_flux_avg[:,d]/injected_flux[d]
            thruput_basis[np.where(thruput_basis < 0)] = 0
            thruput_basis[np.where(thruput_basis > 1)] = 0
        
            for d in range(0, nbr_dist):
                snr_basis_per[:, d, br,i] = [get_snr(frames_basis_fc[n,br,:,:], 
                    loc[d,br,0], loc[d,br,1], fwhm, fmerit)[0] for n in range(0,nnpcs)]
            
        thruput_per[:,:,i] = np.nanmean(thruput_basis[:,:,:], axis = 2)
        
    
    print(thruput_per)
    snr_basis = np.median(snr_basis_per, axis = 2)
    thruput_basis = np.nanmean(thruput_basis[:,:,:], axis = 2)
    print(snr_basis)
    thru_cont_basis = np.zeros((nnpcs, nbr_dist, 3))
    #snr_basis (nnpcs, nbr_dist, per)
    snr_mean = np.mean(snr_basis, axis = 0)
    for d in range(nbr_dist):
        flux_index = np.argmin(snr_mean[d,:]-snr_target)
        thru_cont_basis[:,d,0] = thruput_per[:,d,flux_index]
    
    #Correction on thruput done here
    thru_cont_basis[:,:,0] = thruput_basis
    
    #Try it without this correction, as we use flux close to SNR = 10?...
    for k,n in enumerate(ncomp):
        for d in range(nbr_dist):
            this_curve = thruput_per[k,d,:]
            corrected_thru = interpol(fc_snr[d]*noise_avg[k,d,0], 
                                      all_fluxes[d,:], this_curve)
            
            if corrected_thru < 0 or corrected_thru > 1:
                corrected_thru = 1e-4
            thru_cont_basis[k,d,0] = corrected_thru

    if isinstance(starphot, float) or isinstance(starphot, int):
        thru_cont_basis[:,:,1] = (
            (sigma * noise_avg[:,:,0]) / thru_cont_basis[:,:,0]
        ) / starphot
    else:
        thru_cont_basis[:,:,1] = (
            (sigma * noise_avg[:,:,0]) / thru_cont_basis[:,:,0]
        ) / np.median(starphot)
    if 'multi_epoch' not in algo_name:
        thru_cont_basis[:,:,2] = ncomp.reshape(ncomp.shape[0],1)
    
    
    for d in range(nbr_dist):
        thru_cont_basis[np.where(thru_cont_basis[:,d,0] == 1e-4),d,1] = 1
    print(thru_cont_basis)
    
    
    if through_thresh != 'auto':
        if np.isscalar(through_thresh):
            through_thresh = [through_thresh]*nbr_dist
        for d in range(nbr_dist):
            if np.sum((thru_cont_basis[:,d,0]>through_thresh[d])) == 0:
                print('through_thresh should be at most {}'.format(np.max(thru_cont_basis[:,d,0])))
                raise ValueError('through_thresh is too high')
        
    Optimal_comp_basis = []
    if through_thresh != 'auto':
        for d in range(nbr_dist):
            #pose limit on threshold if it increase with comp...
            sorted_ind = np.argsort(thru_cont_basis[:,d,1])
            for Ind in sorted_ind:
                if thru_cont_basis[Ind,d,0] < through_thresh[d]:
                    continue
                Optimal_comp_basis.append(ncomp[Ind])
                break
            #Optimal_comp_basis.append(ncomp[np.argmin(
            #    thru_cont_basis[thru_cont_basis[:,d,0]>through_thresh[d], d, 1])])
    else:
        for d in range(nbr_dist):
            Optimal_comp_basis.append(ncomp[np.argmin(thru_cont_basis[:, d, 1])])
    Optimal_comp_basis = np.array(Optimal_comp_basis)
    
    if through_thresh == 'auto':
        through_thresh = [thru_cont_basis[np.argmin(thru_cont_basis[:, d, 1]),d,0] 
                                                          for d in range(nbr_dist)]
    elif through_up:
        through_thresh = [thru_cont_basis[np.argmin(thru_cont_basis[
            thru_cont_basis[:,d,0]>through_thresh[d], d, 1]),d,0] for d in range(nbr_dist)]
    
    
    Optimal_comp = np.zeros((nbr_epochs, nbr_dist), dtype = int)
    for i in range(nbr_epochs):
        Optimal_comp[i,:] = Optimal_comp_basis.copy()
        
    Contrast_progress = []
    for d in range(nbr_dist):
        Contrast_progress.append([np.min(thru_cont_basis[:, d, 1])])
        
    Done = []
    I = 0
    while I < iterations:
        Improvements = np.zeros((nbr_epochs,nbr_dist,3))
        
        no_found = np.zeros(nbr_dist)
        for N in range(nbr_epochs):
            
            for d in range(nbr_dist):
                if d in Done:
                    continue
                
                contrasts = np.zeros((nnpcs,2))
                for i, n in enumerate(ncomp):
                    res_cube_a = np.zeros((NbrImages, SizeImage, SizeImage))
                    #res_cube_fc_a = np.zeros((nbranch, NbrImages, SizeImage, SizeImage))
                    
                    these_comp =  []
                    for Nbis in range(nbr_epochs):
                        if Nbis == N:
                            this_comp = i
                        else:
                            this_comp = int(np.where(ncomp == Optimal_comp[Nbis,d])[0])
                        res_cube_a[indices_epochs[Nbis*2]:
                            indices_epochs[(Nbis*2)+1],:,:] = res_cube_no_fc[
                            this_comp,indices_epochs[Nbis*2]:indices_epochs[(Nbis*2)+1],:,:]
                        # res_cube_fc_a[indices_epochs[Nbis*2]:
                        #     indices_epochs[(Nbis*2)+1],:,:] = res_cube_fc[
                        #     this_comp,indices_epochs[Nbis*2]:indices_epochs[(Nbis*2)+1],:,:]
                        these_comp.append(ncomp[this_comp])
                    
                    this_frame = np.median(res_cube_a, axis = 0)
                    #this_frames_fc = np.median(res_cube_fc_a, axis = 1)
                    
                    this_noise, this_mean = noise_dist(this_frame, rad_dist[d], fwhm_med, wedge, 
                                        False, debug)[0]
                    #print(this_noise, this_mean)
                
                    # this_thruput = np.zeros(nbranch)
                    # this_flux = np.array([apertureOne_flux(
                    #     (this_frames_fc[br, :, :] - this_frame), loc[d,br,0], loc[d,br,1], fwhm_med
                    # ) for br in range(nbranch)])
                    
                    # for br in range(nbranch):
                    #     this_thruput[br] = this_flux[br]/all_injected_flux[d, br]
                        
                    # this_thruput[np.where(this_thruput < 0)] = 0
                    # this_thruput[np.where(this_thruput > 1)] = 0
                    
                    # #if np.sum(this_thruput > through_thresh[d]) < nbranch:
                    # #    this_thruput = 0
                    # #print(this_thruput)
                    # if len(np.where(this_thruput > 0)[0]) == 0:
                    #     contrasts[i,0] = 0
                    #     contrasts[i,1] = 1
                    #     continue
                    # else:
                    #     this_avg_thruput = np.nanmean(this_thruput)
                        
                    #Correction on thruput done here
                    #thruput_per = np.zeros((nnpcs, nbr_dist, len(per_values)))
                    #avg_comp = np.mean(these_comp)
                    avg_comp = np.average(these_comp, weights = frames_per_epoch)
                    for k, n in enumerate(ncomp):
                        if n >= avg_comp:
                            n_index = k-1
                            break
                    if avg_comp == np.min(ncomp):
                        n_index = 0
                    curves = thruput_per[n_index:n_index+2,d,:]
                    this_curve = np.zeros((nbr_per))
                    for p in range(0, nbr_per):
                        this_curve[p] = interpol(avg_comp, 
                            [ncomp[n_index], ncomp[n_index+1]], curves[:,p])
                    corrected_thru = interpol(fc_snr[d]*this_noise, 
                            all_fluxes[d,:], this_curve)
                    
                    this_avg_thruput = corrected_thru
                    if this_avg_thruput < through_thresh[d]:
                        contrasts[i,0] = 0
                        contrasts[i,1] = 1
                        continue
                    
                    if isinstance(starphot, float) or isinstance(starphot, int):
                        this_contrast = (
                            (sigma * this_noise) / this_avg_thruput
                        ) / starphot
                    else:
                        this_contrast = (
                            (sigma * this_noise) / this_avg_thruput
                        ) / np.median(starphot)
                    
                    
                    contrasts[i,0] = this_avg_thruput
                    contrasts[i,1] = this_contrast
                #print(contrasts)
                #if np.sum(contrasts == 1) == nnpcs:
                #    continue
                #Optimal_comp[N,d] = ncomp[np.argmin(contrasts)]
                #if np.min(contrasts) < Contrast_progress[int(d)][-1]:
                #    Contrast_progress[d].append(np.min(contrasts))
                
                if np.sum(contrasts[:,1] == 1) == nnpcs:
                    Improvements[N,d,1] = 1
                    Improvements[N,d,0] = Optimal_comp[N,d]
                else:
                    #print(np.min(contrasts[:,1]))
                    BestCompInd = np.argmin(contrasts[:,1])
                    Improvements[N,d,1] = np.min(contrasts[:,1])
                    Improvements[N,d,0] = ncomp[BestCompInd]
                    Improvements[N,d,2] = contrasts[BestCompInd,0]
        
                if approximation == 3:
                    if Improvements[N,d,1] < Contrast_progress[d][-1]:
                        Optimal_comp[N,d] = Improvements[N,d,0]
                        Contrast_progress[d].append(Improvements[N,d,1])
                        if through_up:
                            through_thresh[d] = Improvements[N,d,2]
                        I+=1
                    else:
                        no_found[d] +=1
        
        #print(Improvements)
        for d in range(nbr_dist):
            if d in Done:
                continue
            
            if approximation == 3:
                if no_found[d] == nbr_epochs:
                    Done.append(d)
                    continue
            
            if approximation == 0:
                Best_C = np.min(Improvements[:,d,1])
                if Best_C >= Contrast_progress[int(d)][-1]:
                    Done.append(d)
                    continue
                Best_N = np.argmin(Improvements[:,d,1])
                Optimal_comp[Best_N,d] = Improvements[Best_N,d,0]
                Contrast_progress[d].append(Best_C)
                if through_up:
                    through_thresh[d] = Improvements[Best_N,d,2]
                #print(through_thresh)
                I+=1
                
        #print(Optimal_comp)
        if len(Done) == nbr_dist:
            print('Done after {}'.format(I))
            break
    
    return Optimal_comp, Optimal_comp_basis, Contrast_progress, 




def contrast_multi_epoch_walk3(
    cube,
    angle_list,
    psf_template,
    fwhm,
    distance,
    pxscale,
    starphot,
    algo,
    nbr_cubes,
    step_walk,
    iterations=5,
    through_thresh=0.1,
    through_up=True,
    snr_thresh=0,
    sigma=5,
    nbranch=1,
    theta=0,
    inner_rad=1,
    fc_rad_sep=3,
    approximation = 0,
    switch = 3,
    flux_increase = False,
    noise_sep=1,
    wedge=(0, 360),
    fc_snr=50,
    snr_target=[5,10],
    per_values = [70, 10],
    flux = None,
    opt_ncomp = None,
    cube_delimiter=None,
    cube_ref_delimiter=None,
    epoch_indices=None,
    student=True,
    transmission=None,
    dpi=vip_figdpi,
    debug=False,
    verbose=True,
    full_output=False,
    algo_class=None,
    matrix_adi_ref=None,
    angle_adi_ref=None,
    source_xy = None,
    exclude_negative_lobes=False,
    fmerit = 'mean',
    **algo_dict,
):
    """
    -step_walk: contains the number of image in each epoch. If an integer, all epochs
    will have the same number of images
    
    -nbr_cubes: the number of subdivisions in the processing of the cube
    
    -cube_delimiter: indices containing the limits of each cube to be processed.
    Indices numbered relative to the whole master cube.
    
    -cube_ref_delimiter: indices containing the limits of cube_ref for each cube to be
    processed. Indices numbered relative to the whole master cube.
    
    -epoch_indices: containes limits of each of the subdivisions of each cube to be
    processed. Allows part of a same cube to be processed differently.
    Indices numbered relative to the whole master cube.
    
    -through_thresh:Threshold put on thhe throughput for it to be considered valid
    if through_tresh is 'auto', the threshold is chosen as the throughput of the
    best contrast found in the basis of ncomp tested
    
    -through_up: if True, any contrast value that improves upon the last must also
    increase (or maintain) the throughput to be considered valid
    
    -per_values: percentile values for the fluxes of the fake companions that 
    are injected in the data cube. It then allows for interpolation of the
    throughput for any injected flux
    
    -approximation: Determines the level of approximation to optimize the contrast
    or the snr depending on the mode.
        -1: no approximation. All possibilities tested
        0: test all isolated epochs for improvements, chooses the best
        1: switch from approx = 1 to approx = 3 once a number of steps equal to
        the switch parameter has been done
        2: test all isolated epochs, apply all the best changes at once. Switches
        to approx == 3 once the number of steps done is equal to switch
        3: Test the epochs in order and apply the first best change of component
        it finds
        5:Optimized entirely separately.
        Can then be a tuple, meaning the components found this way will be used
        as starting point for approximation[1] mode. Switch is then applicable
        to this method
    approximation == 0 or -1 only recommended for small datasets, or a small number of 
    epochs. 
    
    -flux: Fix the flux injected for the fake companions. Takes precedence over
    per_values if it is not None.
    
    -iterations: Maximum number of iterations the algorithm will go through
    
    -source_xy : if it is not None, then the function is in snr optimization
    mode. It will try to find the component that increases the snr of the source
    given.
    
    -flux_increase: if source_xy is not None and flux_increase is True, any 
    change of components that increases the snr must also increase the flux of 
    the source.
    
    -opt_ncomp: if flux to inject and optimal ncomp is already known, can skip 
    its calculation by setting per_values to an empyt list and putting the 
    optimal number of principal components in opt_ncomp
    """
    
    from ..metrics import snr
    def get_snr(frame, y, x, fwhm, fmerit):
        """
        """
        if fmerit == 'max':
            yy, xx = disk((y, x), fwhm / 2.)
            res = [snr(frame, (x_, y_), fwhm, plot=False, verbose=False,
                       exclude_negative_lobes=exclude_negative_lobes,
                       full_output=True)
                   for y_, x_ in zip(yy, xx)]
            snr_pixels = np.array(res, dtype=object)[:, -1]
            fluxes = np.array(res, dtype=object)[:, 2]
            argm = np.argmax(snr_pixels)
            # integrated fluxes for the max snr
            return np.max(snr_pixels), fluxes[argm]

        elif fmerit == 'px':
            res = snr(frame, (x, y), fwhm, plot=False, verbose=False,
                      exclude_negative_lobes=exclude_negative_lobes,
                      full_output=True)
            snrpx = res[-1]
            fluxpx = np.array(res, dtype=object)[2]
            # integrated fluxes for the given px
            return snrpx, fluxpx

        elif fmerit == 'mean':
            yy, xx = disk((y, x), fwhm / 2.)
            res = [snr(frame, (x_, y_), fwhm, plot=False, verbose=False,
                       exclude_negative_lobes=exclude_negative_lobes,
                       full_output=True) for y_, x_
                   in zip(yy, xx)]
            snr_pixels = np.array(res, dtype=object)[:, -1]
            fluxes = np.array(res, dtype=object)[:, 2]
            # mean of the integrated fluxes (shifting the aperture)
            return np.mean(snr_pixels), np.mean(fluxes)
        
    def interpol(x, xp, yp):
        xp = np.array(xp)
        yp = np.array(yp)
        indices = np.argsort(xp)
        xp = xp[indices]
        yp = yp[indices]
        if np.isscalar(x):
            x = np.array([x])
        else:
            x = np.array(x)
        for value in x:
            if value < xp[0]:
                slope = (yp[1]-yp[0])/(xp[1]-xp[0])
                return yp[0]+(value-xp[0])*slope
            elif value > xp[-1]:
                slope = (yp[-1]-yp[-2])/(xp[-1]-xp[-2])
                return yp[-1]+(value-xp[-1])*slope
            else:
                for i, v in enumerate(xp):
                    if value > v:
                        continue
                    slope = (yp[i]-yp[i-1])/(xp[i]-xp[i-1])
                    return yp[i-1]+(value-xp[i-1])*slope
        
    #-----------------------------------------------------------------

    if cube.ndim != 3 and cube.ndim != 4:
        raise TypeError("The input array is not a 3d or 4d cube")
    if cube.ndim == 3 and (cube.shape[0] != angle_list.shape[0]):
        raise TypeError("Input parallactic angles vector has wrong length")
    if cube.ndim == 4 and (cube.shape[1] != angle_list.shape[0]):
        raise TypeError("Input parallactic angles vector has wrong length")
    if cube.ndim == 3 and psf_template.ndim != 2:
        raise TypeError("Template PSF is not a frame (for ADI case)")
    if cube.ndim == 4 and psf_template.ndim != 3:
        raise TypeError("Template PSF is not a cube (for ADI+IFS case)")
    if transmission is not None:
        if len(transmission) != 2 and len(transmission) != cube.shape[0] + 1:
            msg = "Wrong shape for transmission should be 2xn_rad or (nch+1) "
            msg += "x n_rad, instead of {}".format(transmission.shape)
            raise TypeError(msg)

    if isinstance(fwhm, (np.ndarray, list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if verbose:
        start_time = time_ini()
        if isinstance(starphot, float) or isinstance(starphot, int):
            msg0 = "ALGO : {}, FWHM = {}, # BRANCHES = {}, SIGMA = {},"
            msg0 += " STARPHOT = {}"
            print(msg0.format(algo.__name__, fwhm_med, nbranch, sigma, starphot))
        else:
            msg0 = "ALGO : {}, FWHM = {}, # BRANCHES = {}, SIGMA = {}"
            print(msg0.format(algo.__name__, fwhm_med, nbranch, sigma))
            
    if cube.ndim != 3 and cube.ndim != 4:
        raise TypeError("The input array is not a 3d or 4d cube")

    else:
        if psf_template.shape[1] % 2 == 0:
            raise ValueError("Only odd-sized PSF is accepted")
        if not hasattr(algo, "__call__"):
            raise TypeError("Parameter `algo` must be a callable function")
        if not isinstance(inner_rad, int):
            raise TypeError("inner_rad must be an integer")
        angular_range = wedge[1] - wedge[0]
        if nbranch > 1 and angular_range < 360:
            msg = "Only a single branch is allowed when working on a wedge"
            raise RuntimeError(msg)
        
    nproc = algo_dict.get("nproc", 1)
    imlib = algo_dict.get("imlib", "vip-fft")
    interpolation = algo_dict.get("interpolation", "lanczos4")
    scaling = algo_dict.get("scaling", None)
    ncomp = algo_dict.get("ncomp")
    
    nnpcs = len(ncomp)
    
    if np.isscalar(ncomp):
        ncomp = np.array([ncomp])
    elif isinstance(ncomp, list):
        ncomp = np.array(ncomp)
            
    algo_name = algo.__name__
    idx = algo.__module__.index('.', algo.__module__.index('.') + 1)
    mod = algo.__module__[:idx]
    tmp = __import__(mod, fromlist=[algo_name.upper()+'_Params'])    
    #algo_params = getattr(tmp, algo_name.upper()+'_Params')
    
    algo_supported = ['pca_annular', 'pca_annular_corr', 
                      'pca_annular_multi_epoch', 'pca_annular_corr_multi_epoch',
                      'pca_annular_mask', 'pca_annular_masked']
    if algo_name not in algo_supported:
        raise ValueError("Algorithm is not supported")
    
    if isinstance(fwhm, (np.ndarray, list)):
        fwhm_med = np.median(fwhm)
    else:
        fwhm_med = fwhm

    if verbose:
        start_time = time_ini()
        
    
    if matrix_adi_ref is not None:
        if 'cube_ref' in algo_dict.keys() and algo_dict['cube_ref'] is not None:
            NAdiRef = algo_dict['cube_ref'].shape[0]
            algo_dict['cube_ref'] = np.vstack((algo_dict['cube_ref'], matrix_adi_ref))
        else:
            NAdiRef = 0
            algo_dict['cube_ref'] = matrix_adi_ref
        NRefT = algo_dict['cube_ref'].shape[0]
    
    if 'annular' in algo_name:
        if isinstance(distance, float):
            distance = np.array([distance])
        elif isinstance(distance, list):
            distance = np.array(distance)
        elif isinstance(distance, np.ndarray):
            pass
        elif distance == 'auto':
            radius_int = algo_dict['radius_int']
            asize = algo_dict['asize']
            y = cube.shape[2]
            n_annuli = int((y / 2 - radius_int) / asize)
            distance = np.array([radius_int+(asize/2) + i*asize for i in range(0, n_annuli)])/fwhm_med
        else:
            raise ValueError("distance parameter must be a float, a list or equal to 'auto'")
            
    SizeImage = int(cube[0].shape[1])
    
    if epoch_indices is not None:
        NbrImages = int(epoch_indices[-1]-epoch_indices[0])
    else:
        NbrImages = int(cube.shape[0])
    
    frames_basis_fc = np.zeros((nnpcs, nbranch, SizeImage, SizeImage), dtype = float)
    frames_basis_no_fc = np.zeros((nnpcs, SizeImage, SizeImage), dtype = float)
    res_cube_fc = np.zeros((nnpcs, nbranch, NbrImages, SizeImage, SizeImage), dtype = float)
    res_cube_no_fc = np.zeros((nnpcs, NbrImages, SizeImage, SizeImage), dtype = float)    
    
    rad_dist = distance * fwhm_med
    nbr_dist = distance.shape[0]
    
    results = []
    algo_dict_copy = algo_dict.copy()
    
    if np.isscalar(step_walk):
        nbr_epochs = int(NbrImages/step_walk)
        step_walk = [step_walk]*nbr_epochs
    else:
        nbr_epochs = len(step_walk)
    
    if isinstance(cube_delimiter, list):
        cube_delimiter = np.array(cube_delimiter)
    if cube_delimiter.shape[0] == nbr_cubes*2:
        R = int(1)
    else:
        R = int(0)

    indices_epochs = np.zeros(nbr_epochs*2, dtype = int)
    
    previous_e = 0
    for n in range(nbr_epochs):
        indices_epochs[n*2:(n*2)+2] = (previous_e, previous_e + step_walk[n])
        previous_e += step_walk[n]
    
    if epoch_indices is not None:
        epoch_saved = np.copy(epoch_indices)
    else:
        epoch_saved = None
        
    for N in range(nbr_cubes):
        algo_dict = algo_dict_copy.copy()
        
        if epoch_indices is not None and len(epoch_indices) == 2 and nbr_cubes > 1:
            algo_dict['epoch_indices'] = epoch_indices
            _, res_cube_no_fc, _ = algo(
                cube=cube, angle_list=angle_list, fwhm=fwhm_med,
                verbose=verbose, full_output = True, **algo_dict)
            epoch_indices = [0]
            for n in range(nbr_epochs):
                epoch_indices.append(epoch_indices[-1]+step_walk[n])
            indices_epochs = [epoch_indices[int(j/2) + j%2] for j in range(0, nbr_epochs*2)]
            break
        
        indices_cube_adi = (cube_delimiter[N+R*N], cube_delimiter[N+R*N+1])
        cube_adi = cube[indices_cube_adi[0]:indices_cube_adi[1]]
        this_angle_list = angle_list[indices_cube_adi[0]:indices_cube_adi[1]]
        if cube_ref_delimiter is not None:
            cube_ref_delimiter = np.array(cube_ref_delimiter)
            Rr = int(0)
            if cube_ref_delimiter.shape[0] == nbr_cubes*2:
                Rr = int(1)
            indices_cube_rdi = (cube_ref_delimiter[N+Rr*N],cube_ref_delimiter[N+Rr*N+1])
            algo_dict['cube_ref'] = algo_dict['cube_ref'][indices_cube_rdi[0]:
                                             indices_cube_rdi[1],:,:]
                
        if algo.__name__ == 'pca_annular_corr':
            if epoch_indices is not None:
                epoch_indices = np.array(epoch_indices)
                Re = int(0)
                if epoch_indices.shape[0] == nbr_cubes*2:
                    Re = int(1)
                algo_dict['epoch_indices'] = epoch_indices[N+Re*N:N+Re*N+2] - indices_cube_adi[0]
            else:
                algo_dict['epoch_indices'] = (indices_cube_adi[0],indices_cube_adi[1]) - indices_cube_adi[0]
            
        if 'delta_rot' in algo_dict.keys():
            if isinstance(algo_dict['delta_rot'], list):
                algo_dict['delta_rot'] = np.array(algo_dict['delta_rot'])
            if isinstance(algo_dict['delta_rot'], np.ndarray):
                if algo_dict['delta_rot'].shape[0] != nbr_cubes:
                    raise ValueError('delta_rot has wrong length')
                algo_dict['delta_rot'] = algo_dict['delta_rot'][N]
        
        
        if algo_name == 'pca_annular_corr':
            indices_done = algo_dict['epoch_indices'] + indices_cube_adi[0]
        else:
            indices_done = np.array(indices_cube_adi)
        
        if 'multi_epoch' not in algo_name:
            _, res_cube_no_fc[:, indices_done[0]:indices_done[1], :, :], _ = algo(
                        cube=cube_adi, angle_list=this_angle_list, fwhm=fwhm_med,
                        verbose=verbose, full_output = True, **algo_dict)
        elif algo_name == 'pca_annular_multi_epoch' or algo_name == 'pca_annular_corr_multi_epoch':
            _, res_cube_no_fc[:, indices_done[0]:indices_done[1], :, :] = algo(
                        cube=cube, angle_list=angle_list, fwhm=fwhm_med,
                        verbose=verbose, full_output = True, **algo_dict)
        else:
            raise ValueError("Algorithm not supported")
          
    frames_per_epoch = [(indices_epochs[i+1]-indices_epochs[i])/NbrImages
                                    for i in range(0, len(indices_epochs), 2)]
    #print(frames_per_epoch)

    frames_basis_no_fc = np.median(res_cube_no_fc, axis = 1)

    noise_avg = np.array([noise_dist(frames_basis_no_fc[n, :, :], rad_dist, fwhm_med, wedge, 
                        False, debug) for n in range(0, nnpcs)])
    
    noise = noise_avg[:, :, 0]
    mean_res = noise_avg[:, :, 1]
    
    indices_epochs = np.array(indices_epochs, dtype = int)
    
    
    #minimizing noise in one half of annulus? look at imapct on source on other side
    
    #add option to kee pSNR above a threshold in contrast optimization
    
    #add approximatin parameters in optimization of contrast
    
    if source_xy is not None:
        
        x_source = source_xy[0]
        y_source = source_xy[1]
        snr_flux_basis = np.array([get_snr(frames_basis_no_fc[i], 
                    y_source, x_source, fwhm, fmerit) for i in range(0, nnpcs)])
        Optimal_comp_basis = ncomp[np.argmax(snr_flux_basis[:, 0])]
        
        Optimal_comp = np.full((nbr_epochs), Optimal_comp_basis)
        
            
        snr_progress = []
        flux_progress = []
        snr_progress.append(np.max(snr_flux_basis[:, 0]))
        flux_progress.append(snr_flux_basis[np.argmax(snr_flux_basis[:, 0]),1])
        
        #Optimize: when more than half values in Imporvements the same,
        #start simply selecting the first one that's better as an iteration
        
        #or a full otimization that always chooses the first better encountered
        #as the iteration
        
        #or step method, group epochs by groups, check all in the group, select the best
        #go to next group, do the same.. Repeat, each change = one iteration
        I = 0
        
        
        if approximation == -1:
            Ncombinations = nnpcs**nbr_epochs
            indices_comp = np.zeros(nbr_epochs, dtype = int)
            
            index = 0
            percentage = 10
            for i in range(Ncombinations):
                if verbose:
                    if i > Ncombinations*(percentage/100) and i < Ncombinations*(percentage/100)+nnpcs:
                        print("Done at {}%".format(percentage))
                        percentage += 10
                res_cube_a = np.zeros((NbrImages, SizeImage, SizeImage))
            
                these_comp =  []
                for Nbis in range(nbr_epochs):
                    this_comp = indices_comp[Nbis]
                    res_cube_a[indices_epochs[Nbis*2]:
                        indices_epochs[(Nbis*2)+1],:,:] = res_cube_no_fc[
                        this_comp,indices_epochs[Nbis*2]:indices_epochs[(Nbis*2)+1],:,:]
                    these_comp.append(ncomp[this_comp])
            
            
                this_frame = np.median(res_cube_a, axis = 0)
                
                this_snr_flux = get_snr(this_frame, 
                                        y_source, x_source, fwhm, fmerit)
            
                if flux_increase:
                    if this_snr_flux[1] < flux_progress[-1]:
                        indices_comp, index = NextComb(indices_comp, nnpcs, nbr_epochs)
                        continue
                    
                if this_snr_flux[0] > snr_progress[-1]:
                    snr_progress.append(this_snr_flux[0])
                    Optimal_comp = these_comp
                        
                    
                indices_comp, index = NextComb(indices_comp, nnpcs, nbr_epochs)
            
            Best_frame_basis = frames_basis_no_fc[np.argmax(snr_flux_basis[:, 0])]
            for Nbis in range(nbr_epochs):
                this_comp = int(np.where(ncomp == Optimal_comp[Nbis])[0])
                res_cube_a[indices_epochs[Nbis*2]:
                    indices_epochs[(Nbis*2)+1],:,:] = res_cube_no_fc[
                    this_comp,indices_epochs[Nbis*2]:indices_epochs[(Nbis*2)+1],:,:]
            Best_frame = np.median(res_cube_a, axis = 0)
            
            return (Optimal_comp, Optimal_comp_basis, snr_flux_basis, snr_progress,
                    flux_progress, Best_frame_basis, Best_frame)
        
        elif approximation == 5 or (isinstance(approximation, tuple) and approximation[0] == 5):
            for N in range(nbr_epochs):
                this_snr_flux=np.array([get_snr(np.median(res_cube_no_fc[n,
                    indices_epochs[N*2]:indices_epochs[(N*2)+1],:,:], axis = 0),
                    y_source, x_source, fwhm, fmerit) for n in range(nnpcs)])
                
                Optimal_comp[N] = ncomp[np.argmax(this_snr_flux[:,0])]
            
            res_cube_a = np.zeros((NbrImages, SizeImage, SizeImage))
            Best_frame_basis = frames_basis_no_fc[np.argmax(snr_flux_basis[:, 0])]
            for Nbis in range(nbr_epochs):
                this_comp = int(np.where(ncomp == Optimal_comp[Nbis])[0])
                res_cube_a[indices_epochs[Nbis*2]:
                    indices_epochs[(Nbis*2)+1],:,:] = res_cube_no_fc[
                    this_comp,indices_epochs[Nbis*2]:indices_epochs[(Nbis*2)+1],:,:]
            Best_frame = np.median(res_cube_a, axis = 0)
            
            final_snr_flux = get_snr(Best_frame, y_source, x_source, fwhm, fmerit)
            snr_progress.append(final_snr_flux[0])
            flux_progress.append(final_snr_flux[1])
            
            if isinstance(approximation, tuple):
                approximation = approximation[1]
            else:
                return (Optimal_comp, Optimal_comp_basis, snr_flux_basis, snr_progress,
                    flux_progress, Best_frame_basis, Best_frame)
            
        
        counter = 0
        while I < iterations:
        #for I in range(iterations):
            Improvements = np.zeros((nbr_epochs,3))
            previous_N = -1
            no_found = 0
            for N in range(nbr_epochs):
                if N == previous_N:
                    continue
                
                snr_flux_test = np.zeros((nnpcs,2))
                for i, n in enumerate(ncomp):
                    res_cube_a = np.zeros((NbrImages, SizeImage, SizeImage))
                    
                    for Nbis in range(nbr_epochs):
                        #optimize: only change section of the epoch N in res_cube_a
                        #reset when Opt_comp has changed
                        if Nbis == N:
                            this_comp = i
                        else:
                            this_comp = int(np.where(ncomp == Optimal_comp[Nbis])[0])
                        res_cube_a[indices_epochs[Nbis*2]:
                            indices_epochs[(Nbis*2)+1],:,:] = res_cube_no_fc[
                            this_comp,indices_epochs[Nbis*2]:indices_epochs[(Nbis*2)+1],:,:]

                    this_frame = np.median(res_cube_a, axis = 0)
                    
                    snr_flux_test[i] = get_snr(this_frame, 
                                            y_source, x_source, fwhm, fmerit)
                
                if flux_increase:
                    sorted_ind = np.argsort(snr_flux_test[:,0])[::-1]
                    for ind in sorted_ind:
                        if snr_flux_test[ind,0] <= snr_progress[-1]:
                            break
                        if snr_flux_test[ind,1] <= flux_progress[-1]:
                            continue
                        Improvements[N,0:2] = snr_flux_test[ind,:]
                        Improvements[N,2] = ncomp[ind]
                        break
                else:
                    BestInd = np.argmax(snr_flux_test[:, 0])
                    Improvements[N,0:2] = snr_flux_test[BestInd,:]
                    Improvements[N,2] = ncomp[BestInd]
                
                if approximation == 3:
                    if Improvements[N,0] > snr_progress[-1]:
                        if flux_increase:
                            #sorted_ind = np.argsort(snr_flux_test[:,0])[::-1]
                            found = False
                            for ind in sorted_ind:
                                if snr_flux_test[ind,0] <= snr_progress[-1]:
                                    break
                                if snr_flux_test[ind,1] <= flux_progress[-1]:
                                    continue
                                found = True
                                I += 1
                                snr_progress.append(snr_flux_test[ind,0])
                                flux_progress.append(snr_flux_test[ind,1])
                                Optimal_comp[N] = ncomp[ind]
                                previous_N = N
                                if verbose:
                                    print(snr_progress[-1])
                                break
                            if found == False:
                                no_found += 1
                        else:
                            I += 1
                            snr_progress.append(Improvements[N,0])
                            Optimal_comp[N] = Improvements[N,2]
                            previous_N = N
                            if verbose:
                                print(snr_progress[-1])
                    else:
                        no_found += 1
                        
            if no_found == nbr_epochs:
                print('Done after {}'.format(I))
                break
            if approximation == 0 or approximation == 1:
                #print(Improvements)
                Best_snr = np.max(Improvements[:,0])
                if Best_snr <= snr_progress[-1]:
                    print('Done after {}'.format(I))
                    break
                
                Best_N = np.argmax(Improvements[:, 0])
                
                if flux_increase:
                    if Improvements[Best_N,1]<=flux_progress[-1]:
                        print('Done after {}'.format(I))
                        break
                
                previous_N = Best_N
                Optimal_comp[Best_N] = Improvements[Best_N, 2]
                snr_progress.append(Best_snr)
                flux_progress.append(Improvements[Best_N,1])
                if verbose:
                    print(snr_progress[-1])
                
                if approximation == 1:
                    counter += 1
                    if counter == switch:
                        print("Switch to approximation == 3")
                        approximation = 3

            elif approximation == 2:
                for N in range(nbr_epochs):
                    if Improvements[N,0] > snr_progress[-1]:
                        Optimal_comp[N] = Improvements[N,2]
                        
                print(Optimal_comp)
                res_cube_a = np.zeros((NbrImages, SizeImage, SizeImage))
                    
                for Nbis in range(nbr_epochs):
                    this_comp = int(np.where(ncomp == Optimal_comp[Nbis])[0])
                    res_cube_a[indices_epochs[Nbis*2]:
                        indices_epochs[(Nbis*2)+1],:,:] = res_cube_no_fc[
                        this_comp,indices_epochs[Nbis*2]:indices_epochs[(Nbis*2)+1],:,:]

                this_frame = np.median(res_cube_a, axis = 0)
                    
                this_snr_flux = get_snr(this_frame, 
                                    y_source, x_source, fwhm, fmerit)
                
                snr_progress.append(this_snr_flux[0])
                flux_progress.append(this_snr_flux[1])
                if verbose:
                    print(snr_progress[-1])
                counter += 1
                if counter == switch:
                    print("Switch to approximation == 3")
                    approximation = 3
            I+=1
        
        Best_frame_basis = frames_basis_no_fc[np.argmax(snr_flux_basis[:, 0])]
        for Nbis in range(nbr_epochs):
            this_comp = int(np.where(ncomp == Optimal_comp[Nbis])[0])
            res_cube_a[indices_epochs[Nbis*2]:
                indices_epochs[(Nbis*2)+1],:,:] = res_cube_no_fc[
                this_comp,indices_epochs[Nbis*2]:indices_epochs[(Nbis*2)+1],:,:]
        Best_frame = np.median(res_cube_a, axis = 0)
        
        return (Optimal_comp, Optimal_comp_basis, snr_flux_basis, snr_progress,
                flux_progress, Best_frame_basis, Best_frame)
    
    
    # We crop the PSF and check if PSF has been normalized (so that flux in
    # 1*FWHM aperture = 1) and fix if needed
    new_psf_size = int(round(3 * fwhm_med))
    if new_psf_size % 2 == 0:
        new_psf_size += 1
    
    n, y, x = cube.shape
    psf_template = normalize_psf(
        psf_template,
        fwhm=fwhm,
        verbose=verbose,
        size=min(new_psf_size, psf_template.shape[1]),
    )
    
    
    # Initialize the fake companions
    angle_branch = angular_range / nbranch
    
    Throughput = np.zeros((nnpcs, nbr_dist, nbranch))
    
    fc_map = np.zeros((y, x))
    cy, cx = frame_center(cube[0])
    parangles = angle_list

    # each branch is computed separately
    if matrix_adi_ref is not None:
        copy_ref = np.copy(algo_dict['cube_ref'])
        
    nbr_per = len(per_values)
    snr_basis_per = np.zeros((nnpcs, nbr_dist, nbranch, nbr_per))
    
    loc = np.zeros((nbr_dist, nbranch, 2))
    thruput_basis = np.zeros((nnpcs, nbr_dist, nbranch))
    recovered_flux_basis = np.zeros((nnpcs, nbr_dist, nbranch))
    all_injected_flux = np.zeros((nbr_dist, nbranch))
    
    if np.isscalar(fc_snr):
        fc_snr = np.array([fc_snr] * nbr_dist)
    elif isinstance(fc_snr, tuple):
        fc_snr = np.linspace(fc_snr[0], fc_snr[1], nbr_dist)
    
    thruput_per = np.zeros((nnpcs, nbr_dist, nbr_per))
    this_flux = flux
    all_fluxes = np.zeros((nbr_dist, nbr_per))
    for i, per in enumerate(per_values):
        
        for br in range(nbranch):
        
            if matrix_adi_ref is not None:
                algo_dict['cube_ref'] = np.copy(copy_ref)
        
            # each pattern is computed separately. For each one the companions
            # are separated by "fc_rad_sep * fwhm", interleaving the injections
            fc_map = np.ones_like(cube[0]) * 1e-6
            fcy = 0
            fcx = 0
        
            if this_flux is None:
                flux = np.array(fc_snr) * np.array([np.percentile(noise_avg[:, d, 0], per) for d in range(0,nbr_dist)])
            
            if br == 0:
                all_fluxes[:,i] = flux
            print(flux)
        
            if matrix_adi_ref is None:
                cube_fc = cube.copy()
            else:
                cube_fc = cube.copy()
                cube_adi_fc = np.copy(algo_dict['cube_ref'][NAdiRef:NRefT, :, :])
                cube_fc = np.vstack((cube_fc, cube_adi_fc))
                parangles = np.concatenate((angle_list, angle_adi_ref))
        
            for d in range(0, nbr_dist):
                cube_fc = cube_inject_companions(
                    cube_fc,
                    psf_template,
                    parangles,
                    flux[d],
                    rad_dists=rad_dist[d],
                    theta=br * angle_branch + theta,
                    nproc=nproc,
                    imlib=imlib,
                    interpolation=interpolation,
                    verbose=False,
                    )
        
            if matrix_adi_ref is not None:
                algo_dict['cube_ref'][NAdiRef:NRefT, :, :] = cube_fc[n:, :, :]
                cube_fc = cube_fc[0:n, :, :]
            
        
            for d in range(0, nbr_dist):
                y = cy + rad_dist[d] * \
                    np.sin(np.deg2rad(br * angle_branch + theta))
                x = cx + rad_dist[d] * \
                    np.cos(np.deg2rad(br * angle_branch + theta))
                fc_map = frame_inject_companion(
                    fc_map, psf_template, y, x, flux[d], imlib, interpolation
                )
                fcy = y
                fcx = x
                loc[d, br ,:] = np.array([fcy, fcx])

            if verbose:
                msg2 = "Fake companions injected in branch {} "
                print(msg2.format(br + 1))
                timing(start_time)
    
            for N in range(nbr_cubes):
                algo_dict = algo_dict_copy.copy()
                
                if epoch_saved is not None  and len(epoch_saved) == 2:
                    algo_dict['epoch_indices'] = epoch_saved
                    _, res_cube_fc[:,br,:,:,:], _ = algo(
                        cube=cube_fc, angle_list=angle_list, fwhm=fwhm_med, 
                        verbose=verbose, full_output = True, **algo_dict)
                    break
                
                indices_cube_adi = (cube_delimiter[N+R*N], cube_delimiter[N+R*N+1])
                cube_adi = cube[indices_cube_adi[0]:indices_cube_adi[1]]
                this_angle_list = angle_list[cube_delimiter[N+R*N]:cube_delimiter[N+R*N+1]]
                if cube_ref_delimiter is not None:
                    cube_ref_delimiter = np.array(cube_ref_delimiter)
                    Rr = int(0)
                    if cube_ref_delimiter.shape[0] == nbr_cubes*2:
                        Rr = int(1)
                    indices_cube_rdi = (cube_ref_delimiter[N+Rr*N],cube_ref_delimiter[N+Rr*N+1])
                    algo_dict['cube_ref'] = algo_dict['cube_ref'][indices_cube_rdi[0]:
                                                 indices_cube_rdi[1],:,:]
                    
                if algo.__name__ == 'pca_annular_corr':
                    if epoch_indices is not None:
                        epoch_indices = np.array(epoch_indices)
                        Re = int(0)
                        if epoch_indices.shape[0] == nbr_cubes*2:
                            Re = int(1)
                        algo_dict['epoch_indices'] = epoch_indices[N+Re*N:N+Re*N+2] - indices_cube_adi[0]
                    else:
                        algo_dict['epoch_indices'] = (indices_cube_adi[0],indices_cube_adi[1]) - indices_cube_adi[0]
                    
                if 'delta_rot' in algo_dict.keys():
                    if isinstance(algo_dict['delta_rot'], list):
                        algo_dict['delta_rot'] = np.array(algo_dict['delta_rot'])
                    if isinstance(algo_dict['delta_rot'], np.ndarray):
                        if algo_dict['delta_rot'].shape[0] != nbr_cubes:
                            raise ValueError('delta_rot has wrong length')
                        algo_dict['delta_rot'] = algo_dict['delta_rot'][N]
                
                
                if algo_name == 'pca_annular_corr':
                    indices_done = algo_dict['epoch_indices'] + indices_cube_adi[0]
                else:
                    indices_done = np.array(indices_cube_adi)
                if 'multi_epoch' not in algo_name:
                    _, res_cube_fc[:,br,indices_done[0]:indices_done[1],:,:], _ = algo(
                        cube=cube_fc, angle_list=angle_list, fwhm=fwhm_med, 
                        verbose=verbose, full_output = True, **algo_dict)
                elif algo_name == 'pca_annular_multi_epoch' or algo_name == 'pca_annular_corr_multi_epoch':
                    _, res_cube_fc[:,br,indices_done[0]:indices_done[1],:,:] = algo(
                        cube=cube_fc, angle_list=angle_list, fwhm=fwhm_med, 
                        verbose=verbose, full_output = True, **algo_dict)
        
            frames_basis_fc[:,br,:,:] = np.median(res_cube_fc[:,br,:,:,:], axis = 1)

            injected_flux = apertureOne_flux(fc_map, loc[:,br,0], loc[:,br,1], fwhm_med)
            
            all_injected_flux[:,br] = injected_flux
            recovered_flux_avg = np.array([apertureOne_flux(
                (frames_basis_fc[n, br, :, :] - frames_basis_no_fc[n, :, :]), loc[:,br,0], loc[:,br,1], fwhm_med
            ) for n in range(0, nnpcs)])
            recovered_flux_basis[:,:,br] = recovered_flux_avg
            for d in range(0, nbr_dist):
                thruput_basis[:,d,br] = recovered_flux_avg[:,d]/injected_flux[d]
            thruput_basis[np.where(thruput_basis < 0)] = 0
            thruput_basis[np.where(thruput_basis > 1)] = 0
        
            for d in range(0, nbr_dist):
                snr_basis_per[:, d, br,i] = [get_snr(frames_basis_fc[n,br,:,:], 
                    loc[d,br,0], loc[d,br,1], fwhm, fmerit)[0] for n in range(0,nnpcs)]
            
        thruput_per[:,:,i] = np.nanmean(thruput_basis[:,:,:], axis = 2)
        
    
    print(thruput_per)
    snr_basis = np.median(snr_basis_per, axis = 2)
    thruput_basis = np.nanmean(thruput_basis[:,:,:], axis = 2)
    print(snr_basis)
    thru_cont_basis = np.zeros((nnpcs, nbr_dist, 3))
    #snr_basis (nnpcs, nbr_dist, per)
    
    flux_wanted = np.zeros((nnpcs, nbr_dist))
    snr_goal = (snr_target[1]+snr_target[0])/2
    if len(per_values) != 0:
        for d in range(nbr_dist):
            for n in range(nnpcs):
                snrmax = np.max(snr_basis[n,d,:])
                snrmin = np.min(snr_basis[n,d,:])
                correct_flux = ((snr_basis[n,d,:] >= snr_target[0]) & (snr_basis[n,d,:] <= snr_target[1])) 
                if np.sum(correct_flux) != 0:
                    ind_f = np.argmin(snr_basis[n,d,:]-snr_goal)
                    flux_wanted[n,d] = all_fluxes[d,ind_f]
                    maxF = np.max(all_fluxes[d,:])
                    minF = np.min(all_fluxes[d,:])
                    flux_int = interpol(snr_goal, snr_basis[n,d,:], all_fluxes[d,:])
                    if flux_int >= minF and flux_int <= maxF:
                        flux_wanted[n,d] = flux_int
                else:
                    maxF = fc_snr[d]*np.max(noise_avg[:,d,0]) * 1
                    minF = fc_snr[d]*np.min(noise_avg[:,d,0]) / 1
                    flux_int = interpol(snr_goal, snr_basis[n,d,:], all_fluxes[d,:])
                    if flux_int >= minF and flux_int <= maxF:
                        flux_wanted[n,d] = flux_int
                    elif snr_target[0] > snrmax:
                        flux_wanted[n,d] = maxF
                    elif snr_target[1] < snrmin:
                        flux_wanted[n,d] = minF
        
            nbr_neg = np.sum((flux_wanted[:,d]<0))
            if nbr_neg > 1:
                raise ValueError('Unable to properly estimate the optimal flux')
            elif nbr_neg == 1:
                ind = np.where(flux_wanted[:,d] < 0)[0]
                if ind == 0:
                    flux_wanted[ind,d] = flux_wanted[ind+1,d]
                elif ind == nbr_dist:
                    flux_wanted[ind,d] = flux_wanted[ind-1,d]
                else:
                    flux_wanted[ind,d] = (flux_wanted[ind+1,d] + flux_wanted[ind-1,d])/2
                
            for k,n in enumerate(ncomp):
                this_curve = thruput_per[k,d,:]
            
                corrected_thru = interpol(flux_wanted[k,d], 
                                      all_fluxes[d,:], this_curve)
            
                if corrected_thru < 0 or corrected_thru > 1:
                    corrected_thru = 1e-4
                thru_cont_basis[k,d,0] = corrected_thru
    else:
        flux_wanted = flux
            
    print(flux_wanted)
    
    if len(per_values) != 0:
        if isinstance(starphot, float) or isinstance(starphot, int):
            thru_cont_basis[:,:,1] = (
                (sigma * noise_avg[:,:,0]) / thru_cont_basis[:,:,0]
            ) / starphot
        else:
            thru_cont_basis[:,:,1] = (
                (sigma * noise_avg[:,:,0]) / thru_cont_basis[:,:,0]
            ) / np.median(starphot)
        if 'multi_epoch' not in algo_name:
            thru_cont_basis[:,:,2] = ncomp.reshape(ncomp.shape[0],1)
    
    
        for d in range(nbr_dist):
            thru_cont_basis[np.where(thru_cont_basis[:,d,0] == 1e-4),d,1] = 1
        print(thru_cont_basis)
    
    
        if through_thresh != 'auto':
            if np.isscalar(through_thresh):
                through_thresh = [through_thresh]*nbr_dist
            for d in range(nbr_dist):
                if np.sum((thru_cont_basis[:,d,0]>through_thresh[d])) == 0:
                    print('through_thresh should be at most {}'.format(np.max(thru_cont_basis[:,d,0])))
                    raise ValueError('through_thresh is too high')
        
        Optimal_comp_basis = []
        Optimal_comp_basis_ind = []
        if through_thresh != 'auto':
            for d in range(nbr_dist):
                #pose limit on threshold if it increases with comp...
                sorted_ind = np.argsort(thru_cont_basis[:,d,1])
                for Ind in sorted_ind:
                    if thru_cont_basis[Ind,d,0] < through_thresh[d]:
                        continue
                    Optimal_comp_basis.append(ncomp[Ind])
                    Optimal_comp_basis_ind.append(Ind)
                    break
        else:
            for d in range(nbr_dist):
                Ind = np.argmin(thru_cont_basis[:, d, 1])
                Optimal_comp_basis.append(ncomp[Ind])
                Optimal_comp_basis_ind.append(Ind)
    else:
        Optimal_comp_basis_ind = []
        Optimal_comp_basis = opt_ncomp
        for d in range(nbr_dist):
            Optimal_comp_basis_ind.append(int(np.where(np.array(ncomp) == opt_ncomp[d])[0]))
        if through_thresh != 'auto':
            if np.isscalar(through_thresh):
                through_thresh = [through_thresh]*nbr_dist
            
    Optimal_comp_basis = np.array(Optimal_comp_basis)
    print(Optimal_comp_basis)
    Optimal_comp_basis_ind = np.array(Optimal_comp_basis_ind, dtype = int)
    print(Optimal_comp_basis_ind)
    
    flux_wanted = np.array(flux_wanted)
    if len(per_values) != 0:
        flux_wanted = [flux_wanted[Optimal_comp_basis_ind[d],d] for d in range(0, nbr_dist)]

    print(flux_wanted)
    
    ########################################################################
    
    snr_basis = np.zeros((nnpcs, nbr_dist, nbranch))
    thruput_basis = np.zeros((nnpcs, nbr_dist, nbranch))
    for br in range(nbranch):
    
        if matrix_adi_ref is not None:
            algo_dict['cube_ref'] = np.copy(copy_ref)
    
        # each pattern is computed separately. For each one the companions
        # are separated by "fc_rad_sep * fwhm", interleaving the injections
        fc_map = np.ones_like(cube[0]) * 1e-6
        fcy = 0
        fcx = 0
    
        flux = flux_wanted
    
        if matrix_adi_ref is None:
            cube_fc = cube.copy()
        else:
            cube_fc = cube.copy()
            cube_adi_fc = np.copy(algo_dict['cube_ref'][NAdiRef:NRefT, :, :])
            cube_fc = np.vstack((cube_fc, cube_adi_fc))
            parangles = np.concatenate((angle_list, angle_adi_ref))
    
        for d in range(0, nbr_dist):
            cube_fc = cube_inject_companions(
                cube_fc,
                psf_template,
                parangles,
                flux[d],
                rad_dists=rad_dist[d],
                theta=br * angle_branch + theta,
                nproc=nproc,
                imlib=imlib,
                interpolation=interpolation,
                verbose=False,
                )
    
        if matrix_adi_ref is not None:
            algo_dict['cube_ref'][NAdiRef:NRefT, :, :] = cube_fc[n:, :, :]
            cube_fc = cube_fc[0:n, :, :]
        
    
        for d in range(0, nbr_dist):
            y = cy + rad_dist[d] * \
                np.sin(np.deg2rad(br * angle_branch + theta))
            x = cx + rad_dist[d] * \
                np.cos(np.deg2rad(br * angle_branch + theta))
            fc_map = frame_inject_companion(
                fc_map, psf_template, y, x, flux[d], imlib, interpolation
            )
            fcy = y
            fcx = x
            loc[d, br ,:] = np.array([fcy, fcx])

        if verbose:
            msg2 = "Fake companions injected in branch {} "
            print(msg2.format(br + 1))
            timing(start_time)

        for N in range(nbr_cubes):
            algo_dict = algo_dict_copy.copy()
            
            if epoch_indices is not None and len(epoch_saved) == 2:
                algo_dict['epoch_indices'] = epoch_saved
                _, res_cube_fc[:,br,:,:,:], _ = algo(
                    cube=cube_fc, angle_list=angle_list, fwhm=fwhm_med, 
                    verbose=verbose, full_output = True, **algo_dict)
                break
            
            indices_cube_adi = (cube_delimiter[N+R*N], cube_delimiter[N+R*N+1])
            cube_adi = cube[indices_cube_adi[0]:indices_cube_adi[1]]
            this_angle_list = angle_list[cube_delimiter[N+R*N]:cube_delimiter[N+R*N+1]]
            if cube_ref_delimiter is not None:
                cube_ref_delimiter = np.array(cube_ref_delimiter)
                Rr = int(0)
                if cube_ref_delimiter.shape[0] == nbr_cubes*2:
                    Rr = int(1)
                indices_cube_rdi = (cube_ref_delimiter[N+Rr*N],cube_ref_delimiter[N+Rr*N+1])
                algo_dict['cube_ref'] = algo_dict['cube_ref'][indices_cube_rdi[0]:
                                             indices_cube_rdi[1],:,:]
                
            if algo.__name__ == 'pca_annular_corr':
                if epoch_indices is not None:
                    epoch_indices = np.array(epoch_indices)
                    Re = int(0)
                    if epoch_indices.shape[0] == nbr_cubes*2:
                        Re = int(1)
                    algo_dict['epoch_indices'] = epoch_indices[N+Re*N:N+Re*N+2] - indices_cube_adi[0]
                else:
                    algo_dict['epoch_indices'] = (indices_cube_adi[0],indices_cube_adi[1]) - indices_cube_adi[0]
                
            if 'delta_rot' in algo_dict.keys():
                if isinstance(algo_dict['delta_rot'], list):
                    algo_dict['delta_rot'] = np.array(algo_dict['delta_rot'])
                if isinstance(algo_dict['delta_rot'], np.ndarray):
                    if algo_dict['delta_rot'].shape[0] != nbr_cubes:
                        raise ValueError('delta_rot has wrong length')
                    algo_dict['delta_rot'] = algo_dict['delta_rot'][N]
            
            
            if algo_name == 'pca_annular_corr':
                indices_done = algo_dict['epoch_indices'] + indices_cube_adi[0]
            else:
                indices_done = np.array(indices_cube_adi)
            if 'multi_epoch' not in algo_name:
                _, res_cube_fc[:,br,indices_done[0]:indices_done[1],:,:], _ = algo(
                    cube=cube_fc, angle_list=angle_list, fwhm=fwhm_med, 
                    verbose=verbose, full_output = True, **algo_dict)
            elif algo_name == 'pca_annular_multi_epoch' or algo_name == 'pca_annular_corr_multi_epoch':
                _, res_cube_fc[:,br,indices_done[0]:indices_done[1],:,:] = algo(
                    cube=cube_fc, angle_list=angle_list, fwhm=fwhm_med, 
                    verbose=verbose, full_output = True, **algo_dict)
    
        frames_basis_fc[:,br,:,:] = np.median(res_cube_fc[:,br,:,:,:], axis = 1)

        injected_flux = apertureOne_flux(fc_map, loc[:,br,0], loc[:,br,1], fwhm_med)
        
        all_injected_flux[:,br] = injected_flux
        recovered_flux_avg = np.array([apertureOne_flux(
            (frames_basis_fc[n, br, :, :] - frames_basis_no_fc[n, :, :]), loc[:,br,0], loc[:,br,1], fwhm_med
        ) for n in range(0, nnpcs)])
        recovered_flux_basis[:,:,br] = recovered_flux_avg
        for d in range(0, nbr_dist):
            thruput_basis[:,d,br] = recovered_flux_avg[:,d]/injected_flux[d]
        thruput_basis[np.where(thruput_basis < 0)] = 0
        thruput_basis[np.where(thruput_basis > 1)] = 0
    
        for d in range(0, nbr_dist):
            snr_basis[:, d, br] = [get_snr(frames_basis_fc[n,br,:,:], 
                loc[d,br,0], loc[d,br,1], fwhm, fmerit)[0] for n in range(0,nnpcs)]
    
    ########################################################################
    print(all_injected_flux)
    print(snr_basis)
    
    new_thruput = np.nanmean(thruput_basis[:,:,:], axis = 2)
    print(new_thruput)
    
    new_thru_cont_basis = np.zeros_like(thru_cont_basis)
    
    new_thru_cont_basis[:,:,0] = new_thruput
    
    if isinstance(starphot, float) or isinstance(starphot, int):
        new_thru_cont_basis[:,:,1] = (
            (sigma * noise_avg[:,:,0]) / new_thru_cont_basis[:,:,0]
        ) / starphot
    else:
        new_thru_cont_basis[:,:,1] = (
            (sigma * noise_avg[:,:,0]) / new_thru_cont_basis[:,:,0]
        ) / np.median(starphot)
        
    print(new_thru_cont_basis)
    
    
    if through_thresh == 'auto':
        through_thresh = [new_thru_cont_basis[np.argmin(new_thru_cont_basis[:, d, 1]),d,0] 
                                                          for d in range(nbr_dist)]
    elif through_up:
        through_thresh = [new_thru_cont_basis[np.argmin(new_thru_cont_basis[
            new_thru_cont_basis[:,d,0]>through_thresh[d], d, 1]),d,0] for d in range(nbr_dist)]
    
    
    Optimal_comp_basis = []
    Optimal_comp_basis_ind = []
    if through_thresh != 'auto':
        for d in range(nbr_dist):
            #pose limit on threshold if it increases with comp...
            sorted_ind = np.argsort(new_thru_cont_basis[:,d,1])
            for Ind in sorted_ind:
                if new_thru_cont_basis[Ind,d,0] < through_thresh[d]:
                    continue
                Optimal_comp_basis.append(ncomp[Ind])
                Optimal_comp_basis_ind.append(Ind)
                break
    else:
        for d in range(nbr_dist):
            Ind = np.argmin(new_thru_cont_basis[:, d, 1])
            Optimal_comp_basis.append(ncomp[Ind])
            Optimal_comp_basis_ind.append(Ind)
    
    Optimal_comp = np.zeros((nbr_epochs, nbr_dist), dtype = int)
    for i in range(nbr_epochs):
        Optimal_comp[i,:] = Optimal_comp_basis.copy()
        
    Contrast_progress = []
    last_thruput = []
    for d in range(nbr_dist):
        Contrast_progress.append([np.min(new_thru_cont_basis[:, d, 1])])
        last_thruput.append(new_thru_cont_basis[np.argmin(new_thru_cont_basis[:, d, 1]),d,0])
        
    if approximation == -1:
        Ncombinations = nnpcs**nbr_epochs
        indices_comp = np.zeros(nbr_epochs, dtype = int)
        
        All_Optimal_Comp = np.array([np.full((nbr_epochs), Optimal_comp_basis[d]) for d in range(nbr_dist)])
        
        index = 0
        I = 0
        while I < Ncombinations:
            res_cube_a = np.zeros((NbrImages, SizeImage, SizeImage))
            res_cube_fc_a = np.zeros((nbranch, NbrImages, SizeImage, SizeImage))
        
            these_comp =  []
            for Nbis in range(nbr_epochs):
                this_comp = indices_comp[Nbis]
                res_cube_a[indices_epochs[Nbis*2]:
                    indices_epochs[(Nbis*2)+1],:,:] = res_cube_no_fc[
                    this_comp,indices_epochs[Nbis*2]:indices_epochs[(Nbis*2)+1],:,:]
                res_cube_fc_a[:,indices_epochs[Nbis*2]:
                    indices_epochs[(Nbis*2)+1],:,:] = res_cube_fc[
                    this_comp,:,indices_epochs[Nbis*2]:indices_epochs[(Nbis*2)+1],:,:]
                these_comp.append(ncomp[this_comp])
        
        
            this_frame = np.median(res_cube_a, axis = 0)
            this_frames_fc = np.median(res_cube_fc_a, axis = 1)
        
            this_noise = noise_dist(this_frame, rad_dist, fwhm_med, wedge, 
                            False, debug)[:,0]
        
        
            this_thruput = np.zeros((nbranch, nbr_dist))
            this_flux = np.array([apertureOne_flux(
                (this_frames_fc[br, :, :] - this_frame), loc[:,br,0], loc[:,br,1], fwhm_med
            ) for br in range(nbranch)])
        
            for br in range(nbranch):
                this_thruput[br,:] = this_flux[br,:]/all_injected_flux[:, br]
            
            this_thruput[np.where(this_thruput < 0)] = 0
            this_thruput[np.where(this_thruput > 1)] = 0
        
            this_contrast = np.zeros((nbr_dist, 2))
        
            for d in range(nbr_dist):
                if len(np.where(this_thruput[:,d] > 0)[0]) == 0:
                    this_contrast[d,0] = 0
                    this_contrast[d,1] = 1
                    continue
                else:
                    this_contrast[d,0] = np.nanmean(this_thruput[:,d], axis = 0)
                
                if this_contrast[d,0] < through_thresh[d]:
                    continue
            
                if isinstance(starphot, float) or isinstance(starphot, int):
                    this_contrast[d,1] = (
                        (sigma * this_noise[d]) / this_contrast[d,0]
                    ) / starphot
                else:
                    this_contrast[d,1] = (
                        (sigma * this_noise[d]) / this_contrast[d,0]
                    ) / np.median(starphot)
                
                
                if this_contrast[d,1] < Contrast_progress[d][-1]:
                    Contrast_progress[d].append(this_contrast[d,1])
                    last_thruput[d] = this_contrast[d,0]
                    All_Optimal_Comp[d,:] = these_comp
                    
            if np.sum(this_contrast[:,0]<through_thresh) == nbr_dist:
                new_index = index
                while new_index == index:
                    I+=1
                    indices_comp, index = NextComb(indices_comp, nnpcs, nbr_epochs)
                continue
            
            indices_comp, index = NextComb(indices_comp, nnpcs, nbr_epochs)
            I+=1
            
        BestFrames = np.zeros((nbr_dist, cube.shape[2], cube.shape[2]))
        for d in range(nbr_dist):
            for Nbis in range(nbr_epochs):
                this_comp = int(np.where(ncomp == Optimal_comp[Nbis,d])[0])
                res_cube_a[indices_epochs[Nbis*2]:
                           indices_epochs[(Nbis*2)+1],:,:] = res_cube_no_fc[
                              this_comp,indices_epochs[Nbis*2]:indices_epochs[(Nbis*2)+1],:,:]
            BestFrames[d] = np.median(res_cube_a, axis = 0)
        
        best_noise_mean = np.array([noise_dist(BestFrames[d], rad_dist[d], fwhm_med, wedge, 
                            False, debug) for d in range(nbr_dist)])
        
        best_noise = best_noise_mean[:,:,0]
        
        LastResult = np.zeros((nbr_dist,3))
        LastResult[:,0] = last_thruput 
        for d in range(0, nbr_dist):
            LastResult[d,1] = Contrast_progress[d][-1]
            
        if student:
            Student_res = np.zeros(nbr_dist)
            n_res_els = np.floor(rad_dist / fwhm_med * 2 * np.pi)
            ss_corr = np.sqrt(1 + 1 / n_res_els)
            sigma_corr = stats.t.ppf(stats.norm.cdf(sigma), n_res_els - 1) * ss_corr
            for d in range(nbr_dist):
                if isinstance(starphot, float) or isinstance(starphot, int):
                    Student_res[d] = (
                        (sigma_corr[d] * best_noise[d]) / last_thruput[d]
                    ) / starphot
                else:
                    Student_res[d] = (
                        (sigma_corr[d] *best_noise[d]) / last_thruput[d]
                    ) / np.median(starphot)
            Student_res[np.where(Student_res < 0)] = 1
            Student_res[np.where(Student_res > 1)] = 1
            LastResult[:,2] = Student_res
        
        return Contrast_progress, All_Optimal_Comp, LastResult
        
    elif approximation == 5:
        for N in range(nbr_epochs):
            if verbose:
                print("Processing epoch {}".format(N+1))
            contrast_values = np.zeros((nbr_dist, nnpcs, 2))
            for n in range(nnpcs):
                this_frame = np.median(res_cube_no_fc[n,
                        indices_epochs[N*2]:indices_epochs[(N*2)+1],:,:], axis = 0)
                this_frames_fc = np.median(res_cube_fc[n,:,
                        indices_epochs[N*2]:indices_epochs[(N*2)+1],:,:], axis = 1)

                this_noise = noise_dist(this_frame, rad_dist, fwhm_med, wedge, 
                            False, debug)[:,0]
        
                this_thruput = np.zeros((nbranch, nbr_dist))
                this_flux = np.array([apertureOne_flux(
                    (this_frames_fc[br, :, :] - this_frame), loc[:,br,0], loc[:,br,1], fwhm_med
                ) for br in range(nbranch)])
                
                for br in range(nbranch):
                    this_thruput[br,:] = this_flux[br,:]/all_injected_flux[:, br]
                
                this_thruput[np.where(this_thruput < 0)] = 0
                this_thruput[np.where(this_thruput > 1)] = 0
                this_contrast = np.zeros((nbr_dist, 2))
                
                for d in range(nbr_dist):
                    this_contrast[d,0] = np.nanmean(this_thruput[:,d], axis = 0)
                    if this_contrast[d,0] < through_thresh[d]:
                        this_contrast[d,1] = 1
                        continue
                
                    if isinstance(starphot, float) or isinstance(starphot, int):
                        this_contrast[d,1] = (
                            (sigma * this_noise[d]) / this_contrast[d,0]
                        ) / starphot
                    else:
                        this_contrast[d,1] = (
                            (sigma * this_noise[d]) / this_contrast[d,0]
                        ) / np.median(starphot)
                    
                contrast_values[:,n,:] = this_contrast
                
            for d in range(nbr_dist):
                Ind = np.argmin(contrast_values[d,:,1])
                last_thruput[d] = contrast_values[d,Ind,0]
                Optimal_comp[N,d] = ncomp[Ind]
        
        res_cube_a = np.zeros_like(cube)
        BestFrames = np.zeros((nbr_dist, cube.shape[2], cube.shape[2]))
        for d in range(nbr_dist):
            for Nbis in range(nbr_epochs):
                this_comp = int(np.where(ncomp == Optimal_comp[Nbis,d])[0])
                res_cube_a[indices_epochs[Nbis*2]:
                           indices_epochs[(Nbis*2)+1],:,:] = res_cube_no_fc[
                              this_comp,indices_epochs[Nbis*2]:indices_epochs[(Nbis*2)+1],:,:]
                
            BestFrames[d] = np.median(res_cube_a, axis = 0)
        
        best_noise_mean = np.array([noise_dist(BestFrames[d], rad_dist[d], fwhm_med, wedge, 
                            False, debug) for d in range(nbr_dist)])
        
        best_noise = best_noise_mean[:,:,0]
        
        return Contrast_progress, Optimal_comp
    
    
    Done = []
    I = 0
    while I < iterations:
        Improvements = np.zeros((nbr_epochs,nbr_dist,3))
        
        no_found = np.zeros(nbr_dist)
        for N in range(nbr_epochs):
            
            if verbose:
                print("Processing epoch {}".format(N+1))
            
            for d in range(nbr_dist):
                if d in Done:
                    continue
                
                contrasts = np.zeros((nnpcs,2))
                for i, n in enumerate(ncomp):
                    res_cube_a = np.zeros((NbrImages, SizeImage, SizeImage))
                    res_cube_fc_a = np.zeros((nbranch, NbrImages, SizeImage, SizeImage))
                    
                    these_comp =  []
                    for Nbis in range(nbr_epochs):
                        if Nbis == N:
                            this_comp = i
                        else:
                            this_comp = int(np.where(ncomp == Optimal_comp[Nbis,d])[0])
                        res_cube_a[indices_epochs[Nbis*2]:
                            indices_epochs[(Nbis*2)+1],:,:] = res_cube_no_fc[
                            this_comp,indices_epochs[Nbis*2]:indices_epochs[(Nbis*2)+1],:,:]
                        res_cube_fc_a[:,indices_epochs[Nbis*2]:
                            indices_epochs[(Nbis*2)+1],:,:] = res_cube_fc[
                            this_comp,:,indices_epochs[Nbis*2]:indices_epochs[(Nbis*2)+1],:,:]
                        these_comp.append(ncomp[this_comp])
                    
                    this_frame = np.median(res_cube_a, axis = 0)
                    this_frames_fc = np.median(res_cube_fc_a, axis = 1)
                    
                    this_noise, this_mean = noise_dist(this_frame, rad_dist[d], fwhm_med, wedge, 
                                        False, debug)[0]
                
                    this_thruput = np.zeros(nbranch)
                    this_flux = np.array([apertureOne_flux(
                        (this_frames_fc[br, :, :] - this_frame), loc[d,br,0], loc[d,br,1], fwhm_med
                    ) for br in range(nbranch)])
                    
                    for br in range(nbranch):
                        this_thruput[br] = this_flux[br]/all_injected_flux[d, br]
                        
                    this_thruput[np.where(this_thruput < 0)] = 0
                    this_thruput[np.where(this_thruput > 1)] = 0
                    
                    # #if np.sum(this_thruput > through_thresh[d]) < nbranch:
                    # #    this_thruput = 0
                    # #print(this_thruput)
                    if len(np.where(this_thruput > 0)[0]) == 0:
                        contrasts[i,0] = 0
                        contrasts[i,1] = 1
                        continue
                    else:
                        this_avg_thruput = np.nanmean(this_thruput)
                        
                    #Correction on thruput done here
                    #thruput_per = np.zeros((nnpcs, nbr_dist, len(per_values)))
                    #avg_comp = np.mean(these_comp)
                    # avg_comp = np.average(these_comp, weights = frames_per_epoch)
                    # for k, n in enumerate(ncomp):
                    #     if n >= avg_comp:
                    #         n_index = k-1
                    #         break
                    # if avg_comp == np.min(ncomp):
                    #     n_index = 0
                    # curves = thruput_per[n_index:n_index+2,d,:]
                    # this_curve = np.zeros((nbr_per))
                    # for p in range(0, nbr_per):
                    #     this_curve[p] = interpol(avg_comp, 
                    #         [ncomp[n_index], ncomp[n_index+1]], curves[:,p])
                    # corrected_thru = interpol(fc_snr[d]*this_noise, 
                    #         all_fluxes[d,:], this_curve)
                    
                    # print(this_avg_thruput, corrected_thru)
                    
                    #this_avg_thruput = corrected_thru
                    if this_avg_thruput < through_thresh[d]:
                        contrasts[i,0] = 0
                        contrasts[i,1] = 1
                        continue
                    
                    if isinstance(starphot, float) or isinstance(starphot, int):
                        this_contrast = (
                            (sigma * this_noise) / this_avg_thruput
                        ) / starphot
                    else:
                        this_contrast = (
                            (sigma * this_noise) / this_avg_thruput
                        ) / np.median(starphot)
                    
                    
                    contrasts[i,0] = this_avg_thruput
                    contrasts[i,1] = this_contrast
                    #print(these_comp)
                    #print(contrasts[i,:])
                #print(contrasts)
                #if np.sum(contrasts == 1) == nnpcs:
                #    continue
                #Optimal_comp[N,d] = ncomp[np.argmin(contrasts)]
                #if np.min(contrasts) < Contrast_progress[int(d)][-1]:
                #    Contrast_progress[d].append(np.min(contrasts))
                
                if np.sum(contrasts[:,1] == 1) == nnpcs:
                    Improvements[N,d,1] = 1
                    Improvements[N,d,0] = Optimal_comp[N,d]
                else:
                    #print(np.min(contrasts[:,1]))
                    BestCompInd = np.argmin(contrasts[:,1])
                    Improvements[N,d,1] = np.min(contrasts[:,1])
                    Improvements[N,d,0] = ncomp[BestCompInd]
                    Improvements[N,d,2] = contrasts[BestCompInd,0]
        
                if approximation == 3:
                    if Improvements[N,d,1] < Contrast_progress[d][-1]:
                        Optimal_comp[N,d] = Improvements[N,d,0]
                        Contrast_progress[d].append(Improvements[N,d,1])
                        last_thruput[d] = Improvements[N,d,2]
                        if through_up:
                            through_thresh[d] = Improvements[N,d,2]
                        #I+=1
                    else:
                        no_found[d] +=1
        
        #print(Improvements)
        for d in range(nbr_dist):
            if d in Done:
                continue
            
            if approximation == 3:
                if no_found[d] == nbr_epochs:
                    Done.append(d)
                    continue
            
            if approximation == 0:
                Best_C = np.min(Improvements[:,d,1])
                if Best_C >= Contrast_progress[int(d)][-1]:
                    Done.append(d)
                    continue
                Best_N = np.argmin(Improvements[:,d,1])
                Optimal_comp[Best_N,d] = Improvements[Best_N,d,0]
                Contrast_progress[d].append(Best_C)
                last_thruput[d] = Improvements[Best_N,d,2]
                if through_up:
                    through_thresh[d] = Improvements[Best_N,d,2]
                #print(through_thresh)
                #I+=1
            
        I+=1
        #print(Optimal_comp)
        if len(Done) == nbr_dist:
            print('Done after {}'.format(I))
            break
        
    BestFrames = np.zeros((nbr_dist, cube.shape[2], cube.shape[2]))
    for d in range(nbr_dist):
        for Nbis in range(nbr_epochs):
            this_comp = int(np.where(ncomp == Optimal_comp[Nbis,d])[0])
            res_cube_a[indices_epochs[Nbis*2]:
                       indices_epochs[(Nbis*2)+1],:,:] = res_cube_no_fc[
                          this_comp,indices_epochs[Nbis*2]:indices_epochs[(Nbis*2)+1],:,:]
        BestFrames[d] = np.median(res_cube_a, axis = 0)
    
    best_noise_mean = np.array([noise_dist(BestFrames[d], rad_dist[d], fwhm_med, wedge, 
                        False, debug) for d in range(nbr_dist)])
    
    best_noise = best_noise_mean[:,:,0]
    
    LastResult = np.zeros((nbr_dist,3))
    LastResult[:,0] = last_thruput 
    for d in range(0, nbr_dist):
        LastResult[d,1] = Contrast_progress[d][-1]
        
    if student:
        Student_res = np.zeros(nbr_dist)
        n_res_els = np.floor(rad_dist / fwhm_med * 2 * np.pi)
        ss_corr = np.sqrt(1 + 1 / n_res_els)
        sigma_corr = stats.t.ppf(stats.norm.cdf(sigma), n_res_els - 1) * ss_corr
        for d in range(nbr_dist):
            if isinstance(starphot, float) or isinstance(starphot, int):
                Student_res[d] = (
                    (sigma_corr[d] * best_noise[d]) / last_thruput[d]
                ) / starphot
            else:
                Student_res[d] = (
                    (sigma_corr[d] *best_noise[d]) / last_thruput[d]
                ) / np.median(starphot)
        Student_res[np.where(Student_res < 0)] = 1
        Student_res[np.where(Student_res > 1)] = 1
        LastResult[:,2] = Student_res
    
    
    return Optimal_comp, Optimal_comp_basis, Contrast_progress, flux_wanted, LastResult




def noise_dist(array, distance, fwhm, wedge=(0, 360), verbose=False, debug=False):
    """
    distance is the distance at which noise level is evaluated
    ATTENTION: in noise per annulus function, separation is the separation 
    between each annuli. Init_rad is the initial distance
    of the first annuli, annuli whose width is equal to separation then...
    
    sep in find_coords is the "angular" separation between each aperture used
    for the evaluation of the noise

    """
    def find_coords(rad, sep, init_angle, fin_angle):
        angular_range = fin_angle - init_angle
        npoints = (np.deg2rad(angular_range) * rad) / sep  # (2*np.pi*rad)/sep
        ang_step = angular_range / npoints  # 360/npoints
        x = []
        y = []
        for i in range(int(npoints)):
            newx = rad * np.cos(np.deg2rad(ang_step * i + init_angle))
            newy = rad * np.sin(np.deg2rad(ang_step * i + init_angle))
            x.append(newx)
            y.append(newy)
        return np.array(y), np.array(x)
    
    init_angle, fin_angle = wedge
    centery, centerx = frame_center(array)
    
    if debug:
        _, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(
            array, origin="lower", interpolation="nearest", alpha=0.5, cmap="gray"
        )


    if not isinstance(distance, np.ndarray):
        distance = np.array([distance])

    noise_mean = np.zeros((distance.shape[0], 2))
    for i, d in enumerate(distance):
        y = centery + d
        rad = dist(centery, centerx, y, centerx)
        yy, xx = find_coords(rad, fwhm, init_angle, fin_angle)
        yy += centery
        xx += centerx

        apertures = CircularAperture(np.array((xx, yy)).T, fwhm / 2)
        fluxes = aperture_photometry(array, apertures)
        fluxes = np.array(fluxes["aperture_sum"])

        noise_mean[i,0] = np.std(fluxes)
        noise_mean[i,1] = np.mean(fluxes)

    if debug:
        for j in range(xx.shape[0]):
            # Circle takes coordinates as (X,Y)
            aper = plt.Circle(
                (xx[j], yy[j]), radius=fwhm / 2, color="r", fill=False, alpha=0.8
            )
            ax.add_patch(aper)
            cent = plt.Circle(
                (xx[j], yy[j]), radius=0.8, color="r", fill=True, alpha=0.5
            )
            ax.add_patch(cent)

    if verbose:
        print("Radius(px) = {}, Noise = {:.3f} ".format(rad, noise_dist))

    return noise_mean


def apertureOne_flux(array, yc, xc, fwhm, ap_factor=1, mean=False, verbose=False):
    """Returns the sum of pixel values in a circular aperture centered on the
    input coordinates. The radius of the aperture is set as (ap_factor*fwhm)/2.

    Parameters
    ----------
    array : numpy ndarray
        Input frame.
    yc, xc : y and x coordinates of sources.
    fwhm : float
        FWHM in pixels.
    ap_factor : int, optional
        Diameter of aperture in terms of the FWHM.

    Returns
    -------
    flux : list of floats
        List of fluxes.

    Note
    ----
    From Photutils documentation, the aperture photometry defines the aperture
    using one of 3 methods:

    'center': A pixel is considered to be entirely in or out of the aperture
              depending on whether its center is in or out of the aperture.
    'subpixel': A pixel is divided into subpixels and the center of each
                subpixel is tested (as above).
    'exact': (default) The exact overlap between the aperture and each pixel is
             calculated.

    """
    flux = 0
    if mean:
        ind = disk((yc, xc), (ap_factor * fwhm) / 2)
        values = array[ind]
        obj_flux = np.mean(values)
    else:
        if isinstance(xc, np.ndarray):
            coords = [(xc[i], yc[i]) for i in range(0, xc.shape[0])]
            aper = CircularAperture(coords, (ap_factor * fwhm) / 2)
        else:
            aper = CircularAperture((xc, yc), (ap_factor * fwhm) / 2)
        obj_flux = aperture_photometry(array, aper, method="exact")
        obj_flux = np.array(obj_flux["aperture_sum"])
        
    flux = obj_flux

    return flux


def NextComb(indices, nnpcs, nbr_epochs):
    
    k=0
    for e in range(nbr_epochs):
        if indices[e] == nnpcs-1:
            indices[e] = 0
        else:
            k = e
            indices[k] += 1
            break
    return indices,k