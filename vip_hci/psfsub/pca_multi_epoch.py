# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 12:16:20 2023

@author: Justin Latour
"""

import numpy as np
from multiprocessing import cpu_count
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
from ..preproc.derotation import _find_indices_adi, _compute_pa_thresh
from ..preproc import cube_rescaling_wavelengths as scwave
from ..preproc import (cube_derotate, cube_collapse, cube_subtract_sky_pca,
                       check_pa_vector, check_scal_vector, cube_crop_frames)
from ..stats import descriptive_stats
from ..var import (frame_center, dist, prepare_matrix, reshape_matrix,
                   cube_filter_lowpass, mask_circle)

from .pca_fullfr import pca, PCA_Params
from .pca_local import (pca_annular, PCA_ANNULAR_Params,
                        pca_annular_corr, PCA_ANNULAR_CORR_Params)


@dataclass
class PCA_MULTI_EPOCH_Params(PCA_Params):
    """
    Set of parameters for the multi-epoch pca
    """
    ncomp : List[int] = 1
    delta_rot : Union[float, List]
    cube_delimiter : List[int] = None
    cube_ref_delimiter : List[int] = None
    
    
@dataclass
class PCA_ANNULAR_MULTI_EPOCH_Params(PCA_ANNULAR_Params):
    """
    Set of parameters for the mutli-epoch annular pca.
    """
    ncomp : List[int] = 1
    delta_rot : Union[float, List]
    cube_delimiter : List[int] = None
    cube_ref_delimiter : List[int] = None
    
@dataclass
class PCA_ANNULAR_CORR_MULTI_EPOCH_Params(PCA_ANNULAR_CORR_Params):
    """
    Set of parameters for the mutli-epoch annular pca.
    """
    ncomp : List[int] = 1
    delta_rot : Union[float, List]
    cube_delimiter : List[int] = None
    cube_ref_delimiter : List[int] = None



def Inherited_Params(algo_params):
    """
    Separate into two dictionnaries the arguments that were inherited and not inherited
    
    If argument in both, it will put in the inherited dictionary
    """
    Attr_class = dir(algo_params)
    Attr_Parent_class = dir(algo_params.__class__.__bases__[0])
    
    NotInherited = {}
    Inherited = {}
    
    for key in Attr_class:
        if key[0] == '_':
            continue
        if key not in Attr_Parent_class:
            NotInherited[key] = getattr(algo_params, key)
        else:
            Inherited[key] = getattr(algo_params, key)
    
    return Inherited, NotInherited


def RemoveKeys(dictionnary, list_keys):
    """
    Remove the keys listed in list_keys from the dictionnary
    """
    ClearedDict = dictionnary.copy()
    for key in list_keys:
        ClearedDict.pop(key, None)
        
    return ClearedDict

    
def pca_multi_epoch(*all_args: list, **all_kwargs: dict):
    """
    ncomp, cube_delimiter and cube_ref_delimiter must be lists!
    
    ncomp : list of number of principal components to be used for each epoch
    
    cube_delimiter : list of indices used to separate the data cube into the 
        different epochs. Each number must be the index of the first image of 
        the epoch, with the following number being the index of the end of the
        epoch(not included), also correponding to the beginning of the next epoch.
        The last number must be the size of the whole datacube.
    
    cube_ref_delimiter : list of indices used to separate the data cube into the 
        different epochs. It can be presented into two formats:
            - the exact same format as for cube_delimiter
            - if some reference images are used in multiple epochs, the first 
            two indiced delimit the index of the start(included) and the 
            end(not included) of the first epoch, then the next two indices 
            do the same for the second epoch, etc.
    """
    
    class_params, rot_options = separate_kwargs_dict(
        initial_kwargs=all_kwargs, parent_class=PCA_MULTI_EPOCH_Params)
    
    
    algo_params = PCA_MULTI_EPOCH_Params(*all_args, **class_params)
    
    #Params_PCA_ME, Params_PCA, rot_options = separate_PCA_ME(all_kwargs)
    #last_kwargs = args_left(algo_params, Params_PCA)
    
    Inherited, NotInherited = Inherited_Params(algo_params)
    
    ToRemove = ['full_output', 'ncomp', 'cube', 'angle_list', 
                'delta_rot', 'weights', 'collapse']
    Args_left = RemoveKeys(Inherited, ToRemove)
    
    NumberEpochs = len(algo_params.ncomp)
    
    if (type(algo_params.delta_rot) == float):
        algo_params.delta_rot = np.full((NumberEpochs), algo_params.delta_rot, dtype = float)
    elif (type(algo_params.delta_rot) == int):
        algo_params.delta_rot = np.full((NumberEpochs), algo_params.delta_rot, dtype = float)
    elif algo_params.delta_rot == None:
        algo_params.delta_rot = np.full((NumberEpochs), algo_params.delta_rot, dtype = None)

    
    Args_left_copy = Args_left.copy()
    GlobalResiduals = np.array([[[]]])
    
    R = int(0)
    if len(algo_params.cube_delimiter) == 2*NumberEpochs:
        R = int(1)
        
    for N in range(0, NumberEpochs, 1):
        Args_left = Args_left_copy.copy()
        
        if algo_params.cube_ref is not None:
            Rr = int(0)
            if len(algo_params.cube_ref_delimiter) == 2*NumberEpochs:
                Rr = int(1)
            #To know the format used for cube_ref_delimiter
            Args_left['cube_ref'] = Args_left['cube_ref'][algo_params.cube_ref_delimiter[N+Rr*N]:
                                             algo_params.cube_ref_delimiter[N+Rr*N+1],:,:]
        
        this_cube = algo_params.cube[algo_params.cube_delimiter[N+R*N]:algo_params.cube_delimiter[N+R*N+1]]
        this_angle_list = algo_params.angle_list[algo_params.cube_delimiter[N+R*N]:algo_params.cube_delimiter[N+R*N+1]]
        
        _, _, _, _, residuals_cube_ = pca(
            this_cube, this_angle_list,
            ncomp = int(algo_params.ncomp[N]), full_output = True, 
            delta_rot = algo_params.delta_rot[N],
            **Args_left, **rot_options)
        
        if N == 0:
            GlobalResiduals = residuals_cube_
        else:
            GlobalResiduals = np.vstack((GlobalResiduals, residuals_cube_))
    
    FinalFrame = cube_collapse(
        GlobalResiduals, mode=algo_params.collapse, w=algo_params.weights
    )

# =============================================================================
#     #RDI case
#     if algo_params.cube_ref is not None:
#         GlobalResiduals = np.array([[[]]])
#         
#         #To know the format used for cube_ref_delimiter
#         ReusedRef = False
#         if len(algo_params.cube_ref_delimiter) == 2*NumberEpochs:
#             ReusedRef = True
#             
#             
#         for i in range(0, NumberEpochs, 1):
#             StartIndexCube = algo_params.cube_delimiter[i]
#             EndIndexCube = algo_params.cube_delimiter[i+1]
#             
#             if ReusedRef:
#                 StartIndexCubeRef = algo_params.cube_ref_delimiter[2*i]
#                 EndIndexCubeRef = algo_params.cube_ref_delimiter[2*i +1]
#             
#             else:
#                 StartIndexCubeRef = algo_params.cube_ref_delimiter[i]
#                 EndIndexCubeRef = algo_params.cube_ref_delimiter[i+1]
#                 
#             _, _, _, _, residuals_cube_ = pca(
#                 algo_params.cube[StartIndexCube:EndIndexCube,:,:],
#                 algo_params.angle_list[StartIndexCube:EndIndexCube],
#                 cube_ref = algo_params.cube_ref[StartIndexCubeRef:EndIndexCubeRef,:,:],
#                 ncomp = int(algo_params.ncomp[i]), full_output = True, 
#                 delta_rot = algo_params.delta_rot[i],
#                 **Args_left, **rot_options)
#                 #**Params_PCA, **last_kwargs, **rot_options)
#                 
#             if i == 0:
#                 GlobalResiduals = residuals_cube_
#             else:
#                 GlobalResiduals = np.vstack((GlobalResiduals, residuals_cube_))
#             
#             #print(GlobalResiduals.shape)
#                 
#             
#         #FinalFrame = np.median(GlobalResiduals, axis = 0)
#         FinalFrame = cube_collapse(
#             GlobalResiduals, mode=algo_params.collapse, w=algo_params.weights
#         )
#         #return FinalFrame
#     
#     
#     #ADI case
#     else:
#         GlobalResiduals = np.array([[[]]])
#         for i in range(0, NumberEpochs, 1):
#             StartIndex = algo_params.cube_delimiter[i]
#             EndIndex = algo_params.cube_delimiter[i+1]
#             
#             _, _, _, _, residuals_cube_ = pca(
#                 algo_params.cube[StartIndex:EndIndex,:,:],
#                 algo_params.angle_list[StartIndex:EndIndex],
#                 ncomp = int(algo_params.ncomp[i]), full_output = True, 
#                 delta_rot = algo_params.delta_rot[i],
#                 **Args_left, **rot_options)
#                 #**Params_PCA, **last_kwargs, **rot_options)
#             
#             if i == 0:
#                 GlobalResiduals = residuals_cube_
#             else:
#                 GlobalResiduals = np.vstack((GlobalResiduals, residuals_cube_))
#         
#         #FinalFrame = np.median(GlobalResiduals, axis = 0)
#         FinalFrame = cube_collapse(
#             GlobalResiduals, mode=algo_params.collapse, w=algo_params.weights
#         )
#         #return FinalFrame
# =============================================================================
    
    if algo_params.full_output:
        return FinalFrame, GlobalResiduals
    else:
        return FinalFrame
    
    
def pca_annular_multi_epoch(*all_args: list, **all_kwargs: dict):
    """
    ncomp, cube_delimiter and cube_ref_delimiter must be lists!
    
    ncomp : list of number of principal components to be used for each epoch
    
    cube_delimiter : list of indices used to separate the data cube into the 
        different epochs. Each number must be the index of the first image of 
        the epoch, with the following number being the index of the end of the
        epoch(not included), also correponding to the beginning of the next epoch.
        The last number must be the size of the whole datacube.
    
    cube_ref_delimiter : list of indices used to separate the data cube into the 
        different epochs. It can be presented into two formats:
            - the exact same format as for cube_delimiter
            - if some reference images are used in multiple epochs, the first 
            two indiced delimit the index of the start(included) and the 
            end(not included) of the first epoch, then the next two indices 
            do the same for the second epoch, etc.
    
    delta_rot: 
        -can be an int, float or None and will then be considered the same
        for all epochs and annulus
        -can be a tuple, will then be the same tuple for all epochs
        -can be a list, of the same length as the number of epochs. Defines
        explicitly the values of delta_rot for each epoch, and it can even be
        different for each annulus.
    """
    
    class_params, rot_options = separate_kwargs_dict(
        initial_kwargs=all_kwargs, parent_class=PCA_ANNULAR_MULTI_EPOCH_Params)
    
    
    algo_params = PCA_ANNULAR_MULTI_EPOCH_Params(*all_args, **class_params)
    
    Inherited, NotInherited = Inherited_Params(algo_params)
    
    ToRemove = ['full_output', 'ncomp', 'cube', 'angle_list',
                'weights', 'collapse']
    Args_left = RemoveKeys(Inherited, ToRemove)
    
    NumberEpochs = len(algo_params.ncomp)
    
    if (type(algo_params.delta_rot) == float):
        algo_params.delta_rot = np.full((NumberEpochs), algo_params.delta_rot, dtype = float)
    elif (type(algo_params.delta_rot) == int):
        algo_params.delta_rot = np.full((NumberEpochs), algo_params.delta_rot, dtype = float)
    elif algo_params.delta_rot == None:
        algo_params.delta_rot = np.full((NumberEpochs), algo_params.delta_rot, dtype = None)
    elif isinstance(algo_params.delta_rot, tuple):
        algo_params.delta_rot = [algo_params.delta_rot]*NumberEpochs
    elif isinstance(algo_params.delta_rot, list) and len(algo_params.delta_rot) != NumberEpochs:
        raise ValueError('Delta_rot must have the same length as the number of epoch if it is a list')
    
    if (type(algo_params.ncomp) == tuple):
        raise TypeError(
            "Ncomp cannot be a tuple in the pca_annular_multi_epoch case."
        )

    
    Args_left_copy = Args_left.copy()
    GlobalResiduals = np.array([[[]]])
    
    R = int(0)
    if len(algo_params.cube_delimiter) == 2*NumberEpochs:
        R = int(1)
    
    for N in range(0, NumberEpochs, 1):
        Args_left = Args_left_copy.copy()
        
        if algo_params.cube_ref is not None:
            Rr = int(0)
            if len(algo_params.cube_ref_delimiter) == 2*NumberEpochs:
                Rr = int(1)
            #To know the format used for cube_ref_delimiter
            Args_left['cube_ref'] = Args_left['cube_ref'][algo_params.cube_ref_delimiter[N+Rr*N]:
                                             algo_params.cube_ref_delimiter[N+Rr*N+1],:,:]
                
        this_cube = algo_params.cube[algo_params.cube_delimiter[N+R*N]:algo_params.cube_delimiter[N+R*N+1]]
        this_angle_list = algo_params.angle_list[algo_params.cube_delimiter[N+R*N]:algo_params.cube_delimiter[N+R*N+1]]
        Args_left['delta_rot'] = algo_params.delta_rot[N]
        
        _, residuals_cube_, _ = pca_annular(
            this_cube, this_angle_list,
            ncomp = algo_params.ncomp[N], full_output = True, 
            **Args_left, **rot_options)
        
        if N == 0:
            GlobalResiduals = residuals_cube_
        else:
            GlobalResiduals = np.vstack((GlobalResiduals, residuals_cube_))
    
    FinalFrame = cube_collapse(
        GlobalResiduals, mode=algo_params.collapse, w=algo_params.weights
    )


# =============================================================================
#     #RDI(+ADI) case
#     if algo_params.cube_ref is not None:
#         GlobalResiduals = np.array([[[]]])
#         
#         #To know the format used for cube_ref_delimiter
#         Rr = int(0)
#         if len(algo_params.cube_ref_delimiter) == 2*NumberEpochs:
#             Rr = int(1)
#             
#             
#         for i in range(0, NumberEpochs, 1):
#             StartIndexCube = algo_params.cube_delimiter[i]
#             EndIndexCube = algo_params.cube_delimiter[i+1]
#             
#             if ReusedRef:
#                 StartIndexCubeRef = algo_params.cube_ref_delimiter[2*i]
#                 EndIndexCubeRef = algo_params.cube_ref_delimiter[2*i +1]
#             
#             else:
#                 StartIndexCubeRef = algo_params.cube_ref_delimiter[i]
#                 EndIndexCubeRef = algo_params.cube_ref_delimiter[i+1]
#                 
#             _, residuals_cube_, _ = pca_annular(
#                 algo_params.cube[StartIndexCube:EndIndexCube,:,:],
#                 algo_params.angle_list[StartIndexCube:EndIndexCube],
#                 cube_ref = algo_params.cube_ref[StartIndexCubeRef:EndIndexCubeRef,:,:],
#                 ncomp = int(algo_params.ncomp[i]), full_output = True, 
#                 delta_rot = algo_params.delta_rot[i],
#                 **Args_left, **rot_options)
#                 
#             if i == 0:
#                 GlobalResiduals = residuals_cube_
#             else:
#                 GlobalResiduals = np.vstack((GlobalResiduals, residuals_cube_))
#                 
#                 
#         #FinalFrame = np.median(GlobalResiduals, axis = 0)
#         FinalFrame = cube_collapse(
#             GlobalResiduals, mode=algo_params.collapse, w=algo_params.weights
#         )
#     
#     #ADI case
#     else:
#         GlobalResiduals = np.array([[[]]])
#         for i in range(0, NumberEpochs, 1):
#             StartIndex = algo_params.cube_delimiter[i]
#             EndIndex = algo_params.cube_delimiter[i+1]
#             
#             _, residuals_cube_, _ = pca_annular(
#                 algo_params.cube[StartIndex:EndIndex,:,:],
#                 algo_params.angle_list[StartIndex:EndIndex],
#                 ncomp = int(algo_params.ncomp[i]), full_output = True, 
#                 delta_rot = algo_params.delta_rot[i],
#                 **Args_left, **rot_options)
#             
#             if i == 0:
#                 GlobalResiduals = residuals_cube_
#             else:
#                 GlobalResiduals = np.vstack((GlobalResiduals, residuals_cube_))
#         
#         #FinalFrame = np.median(GlobalResiduals, axis = 0)
#         FinalFrame = cube_collapse(
#             GlobalResiduals, mode=algo_params.collapse, w=algo_params.weights
#         )
# =============================================================================
    
    if algo_params.full_output:
        return FinalFrame, GlobalResiduals
    else:
        return FinalFrame
    
    
    
def pca_annular_corr_multi_epoch(*all_args: list, **all_kwargs: dict):
    """
    ncomp, cube_delimiter and cube_ref_delimiter must be lists!
    
    ncomp : list of number of principal components to be used for each epoch
    
    cube_delimiter : list of indices used to separate the data cube into the 
        different epochs. Each number must be the index of the first image of 
        the epoch, with the following number being the index of the end of the
        epoch(not included), also correponding to the beginning of the next epoch.
        The last number must be the size of the whole datacube.
    
    cube_ref_delimiter : list of indices used to separate the data cube into the 
        different epochs. It can be presented into two formats:
            - the exact same format as for cube_delimiter
            - if some reference images are used in multiple epochs, the first 
            two indiced delimit the index of the start(included) and the 
            end(not included) of the first epoch, then the next two indices 
            do the same for the second epoch, etc.
            
    delta_rot: 
        -can be an int, float or None and will then be considered the same
        for all epochs and annulus
        -can be a tuple, will then be the same tuple for all epochs
        -can be a list, of the same length as the number of epochs. Defines
        explicitly the values of delta_rot for each epoch, and it can even be
        different for each annulus.
    """
    
    class_params, rot_options = separate_kwargs_dict(
        initial_kwargs=all_kwargs, parent_class=PCA_ANNULAR_CORR_MULTI_EPOCH_Params)
    
    
    algo_params = PCA_ANNULAR_CORR_MULTI_EPOCH_Params(*all_args, **class_params)
    
    Inherited, NotInherited = Inherited_Params(algo_params)
    
    ToRemove = ['full_output', 'ncomp', 'cube', 'angle_list',
                'weights', 'collapse']
    Args_left = RemoveKeys(Inherited, ToRemove)
    
    NumberEpochs = len(algo_params.ncomp)
    
    if (type(algo_params.delta_rot) == float):
        algo_params.delta_rot = np.full((NumberEpochs), algo_params.delta_rot, dtype = float)
    elif (type(algo_params.delta_rot) == int):
        algo_params.delta_rot = np.full((NumberEpochs), algo_params.delta_rot, dtype = float)
    elif algo_params.delta_rot == None:
        algo_params.delta_rot = np.full((NumberEpochs), algo_params.delta_rot, dtype = None)
    elif isinstance(algo_params.delta_rot, tuple):
        algo_params.delta_rot = [algo_params.delta_rot]*NumberEpochs
    elif isinstance(algo_params.delta_rot, list) and len(algo_params.delta_rot) != NumberEpochs:
        raise ValueError('Delta_rot must have the same length as the number of epoch if it is a list')
    
    if (type(algo_params.ncomp) == tuple):
        raise TypeError(
            "Ncomp cannot be a tuple in the pca_annular_multi_epoch case."
        )

    
    Args_left_copy = Args_left.copy()
    GlobalResiduals = np.array([[[]]])
    
    R = int(0)
    if len(algo_params.cube_delimiter) == 2*NumberEpochs:
        R = int(1)
    
    
    for N in range(0, NumberEpochs, 1):
        Args_left = Args_left_copy.copy()
        
        if algo_params.cube_ref is not None:
            Rr = int(0)
            if len(algo_params.cube_ref_delimiter) == 2*NumberEpochs:
                Rr = int(1)
            #To know the format used for cube_ref_delimiter
            Args_left['cube_ref'] = Args_left['cube_ref'][algo_params.cube_ref_delimiter[N+Rr*N]:
                                             algo_params.cube_ref_delimiter[N+Rr*N+1],:,:]
        
        if Args_left['epoch_indices'] is not None:
            Re = int(0)
            if len(algo_params.epoch_indices) == 2*NumberEpochs:
                Re = int(1)
            Args_left['epoch_indices'] = Args_left['epoch_indices'][N+Re*N:N+Re*N+2]
        else:
            Args_left['epoch_indices'] = (algo_params.cube_delimiter[N+R*N],algo_params.cube_delimiter[N+R*N+1])
                
        this_cube = algo_params.cube[algo_params.cube_delimiter[N+R*N]:algo_params.cube_delimiter[N+R*N+1]]
        this_angle_list = algo_params.angle_list[algo_params.cube_delimiter[N+R*N]:algo_params.cube_delimiter[N+R*N+1]]
        Args_left['delta_rot'] = algo_params.delta_rot[N]
        
        _, residuals_cube_, _ = pca_annular_corr(
            this_cube, this_angle_list,
            ncomp = algo_params.ncomp[N], full_output = True, 
            **Args_left, **rot_options)
        
        if N == 0:
            GlobalResiduals = residuals_cube_
        else:
            GlobalResiduals = np.vstack((GlobalResiduals, residuals_cube_))
    
    FinalFrame = cube_collapse(
        GlobalResiduals, mode=algo_params.collapse, w=algo_params.weights
    )
    
    if algo_params.full_output:
        return FinalFrame, GlobalResiduals
    else:
        return FinalFrame