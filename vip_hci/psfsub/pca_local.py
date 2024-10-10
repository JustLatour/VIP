#! /usr/bin/env python
"""
Module with local/smart PCA (annulus or patch-wise in a multi-processing
fashion) model PSF subtraction for ADI, ADI+SDI (IFS) and ADI+RDI datasets.

.. [ABS13]
   | Absil et al. 2013
   | **Searching for companions down to 2 AU from beta Pictoris using the
     L'-band AGPM coronagraph on VLT/NACO**
   | *Astronomy & Astrophysics, Volume 559, Issue 1, p. 12*
   | `https://arxiv.org/abs/1311.4298
     <https://arxiv.org/abs/1311.4298>`_

"""


__author__ = "Carlos Alberto Gomez Gonzalez, Valentin Christiaens, Thomas Bédrine"
__all__ = ["pca_annular", "PCA_ANNULAR_Params", "pca_annular_corr", 
           "PCA_ANNULAR_CORR_Params", "ARDI_DOUBLE_PCA_Params", "ARDI_double_pca"]

import numpy as np
from multiprocessing import cpu_count
from typing import Tuple, List, Union
from enum import Enum
from dataclasses import dataclass
from .svd import get_eigenvectors
from ..preproc import (cube_derotate, cube_collapse, check_pa_vector,
                       check_scal_vector)
from ..preproc import cube_detect_badfr_correlation, cube_crop_frames
from ..preproc import cube_rescaling_wavelengths as scwave
from ..preproc.derotation import _find_indices_adi, _find_indices_adi2, _define_annuli
from ..preproc.rescaling import _find_indices_sdi
from ..config import time_ini, timing
from ..config.paramenum import SvdMode, Imlib, Interpolation, Collapse, ALGO_KEY
from ..config.utils_conf import pool_map, iterable
from ..config.utils_param import setup_parameters, separate_kwargs_dict
from ..stats import descriptive_stats
from ..var import get_annulus_segments, matrix_scaling, mask_circle
from .pca_fullfr import pca, PCA_Params
AUTO = "auto"


@dataclass
class PCA_ANNULAR_Params:
    """
    Set of parameters for the annular PCA module.


    """

    cube: np.ndarray = None
    angle_list: np.ndarray = None
    cube_ref: np.ndarray = None
    scale_list: np.ndarray = None
    radius_int: int = 0
    fwhm: float = 4
    asize: float = 4
    n_segments: Union[int, List[int], AUTO] = 1
    delta_rot: Union[float, Tuple[float], List[float]] = (0.1, 1)
    delta_sep: Union[float, Tuple[float], List[float]] = (0.1, 1)
    ncomp: Union[int, Tuple, np.ndarray, AUTO] = 1
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
    
@dataclass
class PCA_ANNULAR_CORR_Params:
    """
    Set of parameters for the mutli-epoch annular pca.
    """
    cube: np.ndarray = None
    angle_list: np.ndarray = None
    epoch_indices: Union[Tuple[int], List[int]] = None
    cube_ref: np.ndarray = None
    scale_list: np.ndarray = None
    radius_int: int = 0
    fwhm: float = 4
    asize: float = 4
    n_segments: Union[int, List[int], AUTO] = 1
    delta_rot: Union[float, Tuple[float]] = (0.1, 1)
    delta_sep: Union[float, Tuple[float]] = (0.1, 1)
    ncomp: Union[int, Tuple, np.ndarray, AUTO] = 1
    svd_mode: Enum = SvdMode.LAPACK
    nproc: int = 1
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
    step_corr: int = 1
    Mask_Corr: np.ndarray = None
    max_frames_lib: int = None
    min_frames_lib: int = None
    ADI_Lib: Union[int, List[int]] = None
    RDI_Lib: Union[int, List[int]] = None

@dataclass
class ARDI_DOUBLE_PCA_Params:
    """
    Set of parameters for the mutli-epoch annular pca.
    """
    cube: np.ndarray = None
    angle_list: np.ndarray = None
    epoch_indices: Union[Tuple[int], List[int]] = None
    cube_ref: np.ndarray = None
    scale_list: np.ndarray = None
    mask_center_px: int = 0
    radius_int: int = 0
    fwhm: float = 4
    asize: float = 4
    n_segments: Union[int, List[int], AUTO] = 1
    delta_rot: Union[float, Tuple[float]] = (0.1, 1)
    ncomp: Union[int, Tuple] = 1
    svd_mode: Enum = SvdMode.LAPACK
    nproc: int = 1
    tol: float = 1e-1
    scaling: Enum = None
    imlib: Enum = Imlib.VIPFFT
    interpolation: Enum = Interpolation.LANCZOS4
    collapse: Enum = Collapse.MEDIAN
    theta_init: int = 0
    weights: np.ndarray = None
    cube_sig: np.ndarray = None
    full_output: bool = False
    verbose: bool = True
    left_eigv: bool = False
    Step: int = 5
    mask_rdi: np.ndarray = None
    n_annuli: int = None
    Mask_Corr: np.ndarray = None
    ADI_Lib: Union[int, List[int]] = None
    RDI_Lib: Union[int, List[int]] = None
    crop_adi: int = None


def pca_annular(*all_args: List, **all_kwargs: dict):
    """PCA model PSF subtraction for ADI, ADI+RDI or ADI+mSDI (IFS) data.

    The PCA model is computed locally in each annulus (or annular sectors
    according to ``n_segments``). For each sector we discard reference frames
    taking into account a parallactic angle threshold (``delta_rot``) and
    optionally a radial movement threshold (``delta_sep``) for 4d cubes.

    For ADI+RDI data, it computes the principal components from the reference
    library/cube, forcing pixel-wise temporal standardization. The number of
    principal components can be automatically adjusted by the algorithm by
    minimizing the residuals inside each patch/region.

    References: [AMA12]_ for PCA-ADI; [ABS13]_ for PCA-ADI in concentric annuli
    considering a parallactic angle threshold; [CHR19]_ for PCA-ASDI and
    PCA-SADI in one or two steps.

    Parameters
    ----------
    all_args: list, optional
        Positionnal arguments for the PCA annular algorithm. Full list of
        parameters below.
    all_kwargs: dictionary, optional
        Mix of keyword arguments that can initialize a PCA_ANNULAR_Params and
        the optional 'rot_options' dictionnary, with keyword values for
        "border_mode", "mask_val", "edge_blend", "interp_zeros", "ker" (see
        documentation of ``vip_hci.preproc.frame_rotate``). Can also contain a
        PCA_ANNULAR_Params named as `algo_params`.

    PCA annular parameters
    ----------
    cube : numpy ndarray, 3d or 4d
        Input cube.
    angle_list : 1d numpy ndarray
        Vector of derotation angles to align North up in your cube images.
    cube_ref : numpy ndarray, 3d, optional
        Reference library cube. For Reference Star Differential Imaging.
    scale_list : numpy ndarray, 1d, optional
        If provided, triggers mSDI reduction. These should be the scaling
        factors used to re-scale the spectral channels and align the speckles
        in case of IFS data (ADI+mSDI cube). Usually, these can be approximated
        by the last channel wavelength divided by the other wavelengths in the
        cube (more thorough approaches can be used to get the scaling factors,
        e.g. with ``vip_hci.preproc.find_scal_vector``).
    radius_int : int, optional
        The radius of the innermost annulus. By default is 0, if >0 then the
        central circular region is discarded.
    fwhm : float, optional
        Size of the FWHM in pixels. Default is 4.
    asize : float, optional
        The size of the annuli, in pixels.
    n_segments : int or list of ints or 'auto', optional
        The number of segments for each annulus. When a single integer is given
        it is used for all annuli. When set to 'auto', the number of segments is
        automatically determined for every annulus, based on the annulus width.
    delta_rot : float, tuple of floats or list of floats, optional
        Parallactic angle threshold, expressed in FWHM, used to build the PCA
        library. If a tuple of 2 floats is provided, they are used as the lower
        and upper bounds of a linearly increasing threshold as a function of
        separation. If a list is provided, it will correspond to the threshold
        to be adopted for each annulus (length should match number of annuli).
        Default is (0.1, 1), which excludes 0.1 FWHM for the innermost annulus
        up to 1 FWHM for the outermost annulus.
    delta_sep : float, tuple of floats or list of floats, optional
        The radial threshold in terms of the mean FWHM, used to build the PCA
        library (for ADI+mSDI data). If a tuple of 2 floats is provided, they
        are used as the lower and upper bounds of a linearly increasing
        threshold as a function of separation. If a list is provided, it will
        correspond to the threshold to be adopted for each annulus (length
        should match number of annuli). Default is (0.1, 1), which excludes 0.1
        FWHM for the innermost annulus up to 1 FWHM for the outermost annulus.
    ncomp : 'auto', int, tuple/1d numpy array of int, list, tuple of lists, opt
        How many PCs are used as a lower-dimensional subspace to project the
        target (sectors of) frames. Depends on the dimensionality of `cube`.

        * ADI and ADI+RDI (``cube`` is a 3d array): if a single integer is
        provided, then the same number of PCs will be subtracted at each
        separation (annulus). If a tuple is provided, then a different number
        of PCs will be used for each annulus (starting with the innermost
        one). If ``ncomp`` is set to ``auto`` then the number of PCs are
        calculated for each region/patch automatically. If a list of int is
        provided, several npc will be tried at once, but the same value of npc
        will be used for all annuli. If a tuple of lists of int is provided,
        the length of tuple should match the number of annuli and different sets
        of npc will be calculated simultaneously for each annulus, with the
        exact values of npc provided in the respective lists.

        * ADI or ADI+RDI (``cube`` is a 4d array): same input format allowed as
        above, but with a slightly different behaviour if ncomp is a list: if it
        has the same length as the number of channels, each element of the list
        will be used as ``ncomp`` value (whether int, float or tuple) for each
        spectral channel. Otherwise the same behaviour as above is assumed.

        * ADI+mSDI case: ``ncomp`` must be a tuple of two integers or a list of
        tuples of two integers, with the number of PCs obtained from each
        multi-spectral frame (for each sector) and the number of PCs used in the
        second PCA stage (ADI fashion, using the residuals of the first stage).
        If None then the second PCA stage is skipped and the residuals are
        de-rotated and combined.

    svd_mode : Enum, see `vip_hci.config.paramenum.SvdMode`
        Switch for the SVD method/library to be used.
    nproc : None or int, optional
        Number of processes for parallel computing. If None the number of
        processes will be set to (cpu_count()/2).
    min_frames_lib : int, optional
        Minimum number of frames in the PCA reference library.
    max_frames_lib : int, optional
        Maximum number of frames in the PCA reference library. The more
        distant/decorrelated frames are removed from the library.
    tol : float, optional
        Stopping criterion for choosing the number of PCs when ``ncomp``
        is None. Lower values will lead to smaller residuals and more PCs.
    scaling : Enum, see `vip_hci.config.paramenum.Scaling`
        Pixel-wise scaling mode using ``sklearn.preprocessing.scale``
        function. If set to None, the input matrix is left untouched.
    imlib : Enum, see `vip_hci.config.paramenum.Imlib`
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    interpolation :  Enum, see `vip_hci.config.paramenum.Interpolation`
        See the documentation of the ``vip_hci.preproc.frame_rotate`` function.
    collapse : Enum, see `vip_hci.config.paramenum.Collapse`
        Sets the way of collapsing the frames for producing a final image.
    collapse_ifs : Enum, see `vip_hci.config.paramenum.Collapse`
        Sets how spectral residual frames should be combined to produce an
        mSDI image.
    ifs_collapse_range: str 'all' or tuple of 2 int
        If a tuple, it should contain the first and last channels where the mSDI
        residual channels will be collapsed (by default collapses all channels).
    full_output: boolean, optional
        Whether to return the final median combined image only or with other
        intermediate arrays.
    verbose : bool, optional
        If True prints to stdout intermediate info.
    theta_init : int
        Initial azimuth [degrees] of the first segment, counting from the
        positive x-axis counterclockwise (irrelevant if n_segments=1).
    weights: 1d numpy array or list, optional
        Weights to be applied for a weighted mean. Need to be provided if
        collapse mode is 'wmean'.
    cube_sig: numpy ndarray, opt
        Cube with estimate of significant authentic signals. If provided, this
        will be subtracted before projecting cube onto reference cube.

    Returns
    -------
    frame : numpy ndarray, 2d
        [full_output=False] Median combination of the de-rotated cube.
    array_out : numpy ndarray, 3d or 4d
        [full_output=True] Cube of residuals.
    array_der : numpy ndarray, 3d or 4d
        [full_output=True] Cube residuals after de-rotation.
    frame : numpy ndarray, 2d
        [full_output=True] Median combination of the de-rotated cube.
    """
    # Separate parameters of the ParamsObject from the optionnal rot_options
    class_params, rot_options = separate_kwargs_dict(all_kwargs,
                                                     PCA_ANNULAR_Params
                                                     )

    # Extracting the object of parameters (if any)
    algo_params = None
    if ALGO_KEY in rot_options.keys():
        algo_params = rot_options[ALGO_KEY]
        del rot_options[ALGO_KEY]

    if algo_params is None:
        algo_params = PCA_ANNULAR_Params(*all_args, **class_params)

    # by default, interpolate masked area before derotation if a mask is used
    if algo_params.radius_int and len(rot_options) == 0:
        rot_options['mask_val'] = 0
        rot_options['ker'] = 1
        rot_options['interp_zeros'] = True

    global start_time
    if algo_params.verbose:
        start_time = time_ini()

    if algo_params.left_eigv:
        if (
            (algo_params.cube_ref is not None)
            or (algo_params.cube_sig is not None)
            or (algo_params.ncomp == "auto")
        ):
            raise NotImplementedError(
                "left_eigv is not compatible"
                "with 'cube_ref', 'cube_sig', ncomp='auto'"
            )

    # ADI or ADI+RDI data
    if algo_params.cube.ndim == 3:
        if algo_params.verbose:
            add_params = {"start_time": start_time, "full_output": True}
        else:
            add_params = {"full_output": True}

        func_params = setup_parameters(
            params_obj=algo_params, fkt=_pca_adi_rdi, **add_params
        )
        res = _pca_adi_rdi(**func_params, **rot_options)

        cube_out, cube_der, frame = res
        if algo_params.full_output:
            return cube_out, cube_der, frame
        else:
            return frame

    # 4D cube, but no mSDI desired
    elif algo_params.cube.ndim == 4 and algo_params.scale_list is None:
        nch, nz, ny, nx = algo_params.cube.shape
        ifs_adi_frames = np.zeros([nch, ny, nx])
        if not isinstance(algo_params.ncomp, list):
            algo_params.ncomp = [algo_params.ncomp] * nch
        elif len(algo_params.ncomp) != nch:
            algo_params.ncomp = [algo_params.ncomp] * nch
        if np.isscalar(algo_params.fwhm):
            algo_params.fwhm = [algo_params.fwhm] * nch

        cube_out = []
        cube_der = []
        # ADI or RDI in each channel
        for ch in range(nch):
            if algo_params.cube_ref is not None:
                if algo_params.cube_ref[ch].ndim != 3:
                    msg = "Ref cube has wrong format for 4d input cube"
                    raise TypeError(msg)
                cube_ref_tmp = algo_params.cube_ref[ch]
            else:
                cube_ref_tmp = algo_params.cube_ref

            add_params = {
                "cube": algo_params.cube[ch],
                "fwhm": algo_params.fwhm[ch],
                "ncomp": algo_params.ncomp[ch],
                "full_output": True,
                "cube_ref": cube_ref_tmp,
            }

            func_params = setup_parameters(
                params_obj=algo_params, fkt=_pca_adi_rdi, **add_params
            )
            res_pca = _pca_adi_rdi(**func_params, **rot_options)
            cube_out.append(res_pca[0])
            cube_der.append(res_pca[1])
            ifs_adi_frames[ch] = res_pca[-1]

        if algo_params.collapse_ifs is not None:
            frame = cube_collapse(ifs_adi_frames, mode=algo_params.collapse_ifs)
        else:
            frame = ifs_adi_frames

        # convert to numpy arrays
        cube_out = np.array(cube_out)
        cube_der = np.array(cube_der)
        if algo_params.full_output:
            return cube_out, cube_der, frame
        else:
            return frame

    # ADI+mSDI (IFS) datacubes
    elif algo_params.cube.ndim == 4:
        global ARRAY
        ARRAY = algo_params.cube

        z, n, y_in, x_in = algo_params.cube.shape
        algo_params.fwhm = int(np.round(np.mean(algo_params.fwhm)))
        n_annuli = int((y_in / 2 - algo_params.radius_int) / algo_params.asize)

        if np.array(algo_params.scale_list).ndim > 1:
            raise ValueError("Scaling factors vector is not 1d")
        if not algo_params.scale_list.shape[0] == z:
            raise ValueError("Scaling factors vector has wrong length")

        if not isinstance(algo_params.ncomp, tuple):
            msg = "`ncomp` must be a tuple of two integers when "
            msg += "`cube` is a 4d array"
            raise TypeError(msg)
        else:
            ncomp2 = algo_params.ncomp[1]
            algo_params.ncomp = algo_params.ncomp[0]

        if algo_params.verbose:
            print("First PCA subtraction exploiting the spectral variability")
            print("{} spectral channels per IFS frame".format(z))
            print(
                "N annuli = {}, mean FWHM = {:.3f}".format(
                    n_annuli, algo_params.fwhm)
            )

        add_params = {
            "fr": iterable(range(n)),
            "scal": algo_params.scale_list,
            "collapse": algo_params.collapse_ifs,
        }

        func_params = setup_parameters(
            params_obj=algo_params, fkt=_pca_sdi_fr, as_list=True, **add_params
        )
        res = pool_map(
            algo_params.nproc,
            _pca_sdi_fr,
            verbose=algo_params.verbose,
            *func_params,
        )
        residuals_cube_channels = np.array(res)

        # Exploiting rotational variability
        if algo_params.verbose:
            timing(start_time)
            print("{} ADI frames".format(n))

        if ncomp2 is None:
            if algo_params.verbose:
                print("Skipping the second PCA subtraction")

            cube_out = residuals_cube_channels
            cube_der = cube_derotate(
                cube_out,
                algo_params.angle_list,
                nproc=algo_params.nproc,
                imlib=algo_params.imlib,
                interpolation=algo_params.interpolation,
                **rot_options,
            )
            frame = cube_collapse(
                cube_der, mode=algo_params.collapse, w=algo_params.weights
            )

        else:
            if algo_params.verbose:
                print("Second PCA subtraction exploiting angular variability")

            add_params = {
                "cube": residuals_cube_channels,
                "ncomp": ncomp2,
                "cube_ref": None,
            }

            func_params = setup_parameters(
                params_obj=algo_params, fkt=_pca_adi_rdi, **add_params
            )
            res = _pca_adi_rdi(**func_params, **rot_options)

            if algo_params.full_output:
                cube_out, cube_der, frame = res
            else:
                frame = res

        if algo_params.full_output:
            return cube_out, cube_der, frame
        else:
            return frame

    else:
        raise TypeError("Input array is not a 4d or 3d array")
        
        
        
def pca_annular_corr(*all_args: List, **all_kwargs: dict):
    """Similar to pca_annular but more flexible
    
    -ADI_Lib:The number of ADI frames kept in the library to construct the 
        principal components. They are selected based on which images are the most
        correlated to the considered frame. If set to None, no limit on the number
        is applied
        The argument max_frames_lib is applied BEFORE the selection of the most
        correlated ADI frames, therefore, ADI_Lib should be smaller than
        max_frames_lib or equal to None
    -RDI_Lib:Same as ADI_Lib but for the number of RDI images kept.
    -epoch_indices:the frames of the cube on which the pca is actually applied.
        This allows to have a big ADI cubes with some frames used as reference
        images for the principal components calculation only.
    -step_corr:The number of images at once that are processed. To optimize the
        time taken by the algorithm, this parameter can be set to more than one.
        If equal to two for example, pca will be applied on the first two frames
        at once, meaning they will both have the same principal components. 
        ADI_Lib and RDI_Lib are applied on the median of the two images then.
    -Mask_Corr:Is a boolean mask. Normally, the most correlated frames are 
        calculated on the annulus considered for the pca. However, the correlation 
        factors could be influenced by the presence of companions. 
        Mask_Corr allows for the selection of the most correlated frames on a 
        different area of the images. This area will be the same for all the annuli.

    Returns
    -------
    frame : numpy ndarray, 2d
        [full_output=False] Median combination of the de-rotated cube.
    array_out : numpy ndarray, 3d or 4d
        [full_output=True] Cube of residuals.
    array_der : numpy ndarray, 3d or 4d
        [full_output=True] Cube residuals after de-rotation.
    frame : numpy ndarray, 2d
        [full_output=True] Median combination of the de-rotated cube.
    """
    # Separating the parameters of the ParamsObject from the optionnal rot_options
    class_params, rot_options = separate_kwargs_dict(initial_kwargs=all_kwargs,
                                                     parent_class=PCA_ANNULAR_CORR_Params
                                                     )

    # Extracting the object of parameters (if any)
    algo_params = None
    if ALGO_KEY in rot_options.keys():
        algo_params = rot_options[ALGO_KEY]
        del rot_options[ALGO_KEY]

    if algo_params is None:
        algo_params = PCA_ANNULAR_CORR_Params(*all_args, **class_params)

    # by default, interpolate masked area before derotation if a mask is used
    if algo_params.radius_int and len(rot_options) == 0:
        rot_options['mask_val'] = 0
        rot_options['ker'] = 1
        rot_options['interp_zeros'] = True
        
    if algo_params.epoch_indices == None:
        algo_params.epoch_indices = (0, algo_params.cube.shape[0])
        

    global start_time
    start_time = time_ini()

    if algo_params.left_eigv:
        if (
            (algo_params.cube_ref is not None)
            or (algo_params.cube_sig is not None)
            or (algo_params.ncomp == "auto")
        ):
            raise NotImplementedError(
                "left_eigv is not compatible"
                "with 'cube_ref', 'cube_sig', ncomp='auto'"
            )

    # ADI or ADI+RDI data
    if algo_params.cube.ndim == 3:
        add_params = {"start_time": start_time, "full_output": True}
        
        NbrImages = algo_params.epoch_indices[1]-algo_params.epoch_indices[0]
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
        
        
def ARDI_double_pca(*all_args: List, **all_kwargs: dict):
    
    class_params, rot_options = separate_kwargs_dict(initial_kwargs=all_kwargs,
                                    parent_class=ARDI_DOUBLE_PCA_Params)
    
    algo_params = ARDI_DOUBLE_PCA_Params(*all_args, **class_params)
    
    if not (isinstance(algo_params.ncomp, tuple) or  (isinstance(algo_params.ncomp, int))):
         raise TypeError("ncomp must be a tuple or an int")
    
    if not isinstance(algo_params.ncomp, tuple):
        ncomp = (algo_params.ncomp, algo_params.ncomp)
    else:
        ncomp = algo_params.ncomp

    pca_dir = dir(PCA_Params)
    pca_kwargs = {}
    for key in dir(algo_params):
        if key[0] == '_':
            continue
        if (key == 'ncomp') or (key == 'full_output') or (key == 'delta_rot'):
            continue
        if key not in pca_dir:
            continue
        pca_kwargs[key] = getattr(algo_params, key)
        
    Result_RDI = pca(**pca_kwargs, ncomp = ncomp[0], full_output = True, **rot_options)
    _, _, _, residuals_adi, _ = Result_RDI
    """
    Result_RDI = pca(algo_params.cube,
                     algo_params.angle_list,
                     algo_params.cube_ref,
                     ncomp = ncomp[0],
                     svd_mode = algo_params.svd_mode,
                     imlib = algo_params.imlib,
                     interpolation = algo_params.interpolation,
                     collapse = algo_params.collapse,
                     mask_rdi = algo_params.mask_rdi,
                     nproc = algo_params.nproc,
                     full_output = True, 
                     weights = algo_params.weights)
    """
    
    if algo_params.crop_adi is not None:
        residuals_adi = cube_crop_frames(residuals_adi, int(algo_params.crop_adi), 
                                         verbose = False, force = True)
        
    if algo_params.epoch_indices is None:
        pca_annular_dir = dir(PCA_ANNULAR_Params)
    else:
        pca_annular_dir = dir(PCA_ANNULAR_CORR_Params)
        
    pca_annular_kwargs = {}
    for key in dir(algo_params):
        if key[0] == '_':
            continue
        if (key == 'ncomp' or key == 'cube_ref'):
            continue
        if key not in pca_annular_dir:
            continue
        pca_annular_kwargs[key] = getattr(algo_params, key)

    
    if ((algo_params.mask_rdi is not None) and isinstance(algo_params.mask_rdi, tuple)
                and (algo_params.n_annuli is not None)):
        mask_copy = np.copy(algo_params.mask_rdi[1])
        yc = int(mask_copy.shape[1]/2)
        start = False
        Crop = True
        for i in range(0, yc, 1):
            pixel = mask_copy[yc, yc+i]
            if pixel == 0 and start == False:
                continue
            elif (start == False) and (pixel == 1):
                radius_int = i
                start = True
                continue
            elif pixel == 0:
                width = i - radius_int
                break
            elif i == yc-1:
                width = i - radius_int
        
        asize = int(width / algo_params.n_annuli)
        
        pca_annular_kwargs['radius_int'] = radius_int
        pca_annular_kwargs['asize'] = asize        
        
        if (yc - asize) < radius_int + width:
            Crop = False
            
        if 'Mask_Corr' in pca_annular_kwargs and pca_annular_kwargs['Mask_Corr'] is not None:
            mask_corr_copy = np.copy(algo_params.Mask_Corr)
            y_c = int(mask_corr_copy.shape[1]/2)
            size = mask_corr_copy.shape[1]
            for i in range(size-1, 0, -1):
                pixel = mask_corr_copy[y_c, i]
                if pixel == 0:
                    continue
                else:
                    crop_size_corr = i - y_c
                    break
        else:
            crop_size_corr = 0
        
        if Crop == True:
            if algo_params.cube.shape[2] % 2 == 1:
                crop_size = (radius_int + width) * 2 + 3
            else:
                crop_size = (radius_int + width) * 2 + 2
            crop_size =int(np.max(crop_size, crop_size_corr))
            residuals_adi = cube_crop_frames(residuals_adi, crop_size, 
                                             verbose = False)
    
    pca_annular_kwargs['cube'] = residuals_adi
    
    if algo_params.epoch_indices is None:
        Final_Result = pca_annular(**pca_annular_kwargs, ncomp = ncomp[1], **rot_options)
    else:
        Final_Result = pca_annular_corr(**pca_annular_kwargs, ncomp = ncomp[1], **rot_options)
    
    if algo_params.full_output == False and Crop == True:
        mask_return = np.ones_like(algo_params.cube[0, :, :], dtype = bool)
        mask_return = mask_circle(mask_return, int(crop_size/2), mode = 'out')
        mask_inter = np.copy(mask_return)
        mask_inter = mask_inter[np.newaxis, :, :]
        mask_inter = cube_crop_frames(mask_inter, crop_size, verbose = False)
        mask_inter = mask_inter.reshape((crop_size, crop_size))
        
        if len(ncomp[1]) == 1:
            Final_Result = Final_Result[np.newaxis, :, :]
        
        Final_Result = np.array(Final_Result)
        Result = np.zeros((len(ncomp[1]), algo_params.cube.shape[1], algo_params.cube.shape[1]))
        Result[:, mask_return] = Final_Result[:, mask_inter]
        
        if len(ncomp[1]) == 1:
            Result = Result.reshape((algo_params.cube.shape[1], algo_params.cube.shape[1]))
            
        return Result
    
    return Final_Result
        

################################################################################
# Functions encapsulating portions of the main algorithm
################################################################################


def _pca_sdi_fr(
    fr,
    scal,
    radius_int,
    fwhm,
    asize,
    n_segments,
    delta_sep,
    ncomp,
    svd_mode,
    tol,
    scaling,
    imlib,
    interpolation,
    collapse,
    ifs_collapse_range,
    theta_init,
):
    """Optimized PCA subtraction on a multi-spectral frame (IFS data)."""
    z, n, y_in, x_in = ARRAY.shape

    scale_list = check_scal_vector(scal)
    # rescaled cube, aligning speckles
    multispec_fr = scwave(
        ARRAY[:, fr, :, :], scale_list, imlib=imlib, interpolation=interpolation
    )[0]

    # Exploiting spectral variability (radial movement)
    fwhm = int(np.round(np.mean(fwhm)))
    n_annuli = int((y_in / 2 - radius_int) / asize)

    if isinstance(n_segments, int):
        n_segments = [n_segments for _ in range(n_annuli)]
    elif n_segments == "auto":
        n_segments = list()
        n_segments.append(2)  # for first annulus
        n_segments.append(3)  # for second annulus
        ld = 2 * np.tan(360 / 4 / 2) * asize
        for i in range(2, n_annuli):  # rest of annuli
            radius = i * asize
            ang = np.rad2deg(2 * np.arctan(ld / (2 * radius)))
            n_segments.append(int(np.ceil(360 / ang)))

    cube_res = np.zeros_like(multispec_fr)  # shape (z, resc_y, resc_x)

    if isinstance(delta_sep, tuple):
        delta_sep_vec = np.linspace(delta_sep[0], delta_sep[1], n_annuli)
    elif np.isscalar(delta_sep):
        delta_sep_vec = [delta_sep] * n_annuli
    else:
        if len(delta_sep) != n_annuli:
            msg = "If delta_sep is a list it should have n_annuli elements."
            raise TypeError(msg)
        delta_sep_vec = delta_sep

    for ann in range(n_annuli):
        if ann == n_annuli - 1:
            inner_radius = radius_int + (ann * asize - 1)
        else:
            inner_radius = radius_int + ann * asize
        ann_center = inner_radius + (asize / 2)

        indices = get_annulus_segments(
            multispec_fr[0], inner_radius, asize, n_segments[ann], theta_init
        )
        # Library matrix is created for each segment and scaled if needed
        for seg in range(n_segments[ann]):
            yy = indices[seg][0]
            xx = indices[seg][1]
            matrix = multispec_fr[:, yy, xx]  # shape (z, npx_annsegm)
            matrix = matrix_scaling(matrix, scaling)

            for j in range(z):
                indices_left = _find_indices_sdi(
                    scal, ann_center, j, fwhm, delta_sep_vec[ann]
                )
                matrix_ref = matrix[indices_left]
                curr_frame = matrix[j]  # current frame
                V = get_eigenvectors(
                    ncomp,
                    matrix_ref,
                    svd_mode,
                    noise_error=tol,
                    debug=False,
                    scaling=scaling,
                )
                transformed = np.dot(curr_frame, V.T)
                reconstructed = np.dot(transformed.T, V)
                residuals = curr_frame - reconstructed
                # return residuals, V.shape[0], matrix_ref.shape[0]
                cube_res[j, yy, xx] = residuals

    if ifs_collapse_range == "all":
        idx_ini = 0
        idx_fin = z
    else:
        idx_ini = ifs_collapse_range[0]
        idx_fin = ifs_collapse_range[1]

    frame_desc = scwave(
        cube_res[idx_ini:idx_fin],
        scale_list[idx_ini:idx_fin],
        full_output=False,
        inverse=True,
        y_in=y_in,
        x_in=x_in,
        imlib=imlib,
        interpolation=interpolation,
        collapse=collapse,
    )
    return frame_desc


def _pca_adi_rdi(
    cube,
    angle_list,
    radius_int=0,
    fwhm=4,
    asize=2,
    n_segments=1,
    delta_rot=1,
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
    """PCA exploiting angular variability (ADI fashion)."""
    array = cube
    if array.ndim != 3:
        raise TypeError("Input array is not a cube or 3d array")
    if array.shape[0] != angle_list.shape[0]:
        raise TypeError("Input vector or parallactic angles has wrong length")

    n, y, x = array.shape

    angle_list = check_pa_vector(angle_list)
    n_annuli = int((y / 2 - radius_int) / asize)

    if isinstance(delta_rot, tuple):
        delta_rot = np.linspace(delta_rot[0], delta_rot[1], num=n_annuli)
    elif np.isscalar(delta_rot):
        delta_rot = [delta_rot] * n_annuli
    else:
        if len(delta_rot) != n_annuli:
            msg = "If delta_rot is a list it should have n_annuli elements."
            raise TypeError(msg)


    if isinstance(n_segments, int):
        n_segments = [n_segments for _ in range(n_annuli)]
    elif n_segments == "auto":
        n_segments = list()
        n_segments.append(2)  # for first annulus
        n_segments.append(3)  # for second annulus
        ld = 2 * np.tan(360 / 4 / 2) * asize
        for i in range(2, n_annuli):  # rest of annuli
            radius = i * asize
            ang = np.rad2deg(2 * np.arctan(ld / (2 * radius)))
            n_segments.append(int(np.ceil(360 / ang)))

    if verbose:
        msg = "N annuli = {}, FWHM = {:.3f}"
        print(msg.format(n_annuli, fwhm))
        print("PCA per annulus (or annular sectors):")

    if nproc is None:  # Hyper-threading "duplicates" the cores -> cpu_count/2
        nproc = cpu_count() // 2

    # The annuli are built, and the corresponding PA thresholds for frame
    # rejection are calculated (at the center of the annulus)
    cube_out = np.zeros_like(array)
    if isinstance(ncomp, list):
        nncomp = len(ncomp)
        cube_out = np.zeros([nncomp, array.shape[0], array.shape[1],
                             array.shape[2]])
    if verbose:
        #  verbosity set to 2 only for ADI
        verbose_ann = int(verbose) + int(cube_ref is None)
    else:
        verbose_ann = verbose

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

        n_segments_ann = n_segments[ann]
        res_ann_par = _define_annuli(
            angle_list,
            ann,
            n_annuli,
            fwhm,
            radius_int,
            asize,
            delta_rot[ann],
            n_segments_ann,
            verbose_ann,
            True,
        )
        pa_thr, inner_radius, ann_center = res_ann_par
        indices = get_annulus_segments(
            array[0], inner_radius, asize, n_segments_ann, theta_init
        )

        if left_eigv:
            indices_out = get_annulus_segments(array[0], inner_radius, asize,
                                               n_segments_ann, theta_init,
                                               out=True)

        # Library matrix is created for each segment and scaled if needed
        for j in range(n_segments_ann):
            yy = indices[j][0]
            xx = indices[j][1]
            matrix_segm = array[:, yy, xx]  # shape [nframes x npx_segment]
            matrix_segm = matrix_scaling(matrix_segm, scaling)
            if cube_ref is not None:
                matrix_segm_ref = cube_ref[:, yy, xx]
                matrix_segm_ref = matrix_scaling(matrix_segm_ref, scaling)
            else:
                matrix_segm_ref = None
            if cube_sig is not None:
                matrix_sig_segm = cube_sig[:, yy, xx]
            else:
                matrix_sig_segm = None

            if not left_eigv:
                res = pool_map(
                    nproc,
                    do_pca_patch,
                    matrix_segm,
                    iterable(range(n)),
                    angle_list,
                    fwhm,
                    pa_thr,
                    ann_center,
                    svd_mode,
                    ncompann,
                    min_frames_lib,
                    max_frames_lib,
                    tol,
                    matrix_segm_ref,
                    matrix_sig_segm,
                )

                if isinstance(ncomp, list):
                    nncomp = len(ncomp)
                    residuals = []
                    for nn in range(nncomp):
                        tmp = np.array([res[i][0][nn] for i in range(n)])
                        residuals.append(tmp)
                else:
                    res = np.array(res, dtype=object)
                    residuals = np.array(res[:, 0])
                    ncomps = res[:, 1]
                    nfrslib = res[:, 2]
            else:
                yy_out = indices_out[j][0]
                xx_out = indices_out[j][1]
                matrix_out_segm = array[
                    :, yy_out, xx_out
                ]  # shape [nframes x npx_out_segment]
                matrix_out_segm = matrix_scaling(matrix_out_segm, scaling)
                if isinstance(ncomp, list):
                    npc = max(ncomp)
                else:
                    npc = ncomp
                V = get_eigenvectors(npc, matrix_out_segm, svd_mode,
                                     noise_error=tol, left_eigv=True)

                if isinstance(ncomp, list):
                    residuals = []
                    for nn, npc_tmp in enumerate(ncomp):
                        transformed = np.dot(V[:npc_tmp], matrix_segm)
                        reconstructed = np.dot(transformed.T, V[:npc_tmp])
                        residuals.append(matrix_segm - reconstructed.T)
                else:
                    transformed = np.dot(V, matrix_segm)
                    reconstructed = np.dot(transformed.T, V)
                    residuals = matrix_segm - reconstructed.T
                    nfrslib = matrix_out_segm.shape[0]

            if isinstance(ncomp, list):
                for nn, npc in enumerate(ncomp):
                    for fr in range(n):
                        cube_out[nn, fr][yy, xx] = residuals[nn][fr]
            else:
                for fr in range(n):
                    cube_out[fr][yy, xx] = residuals[fr]

            # number of frames in library printed for each annular quadrant
            # number of PCs printed for each annular quadrant
            if verbose == 2 and not isinstance(ncomp, list):
                descriptive_stats(nfrslib, verbose=verbose, label="\tLIBsize: ")
                descriptive_stats(ncomps, verbose=verbose, label="\tNum PCs: ")

        if verbose == 1:
            print("Done PCA with {} for current annulus".format(svd_mode))
            timing(start_time)

    if isinstance(ncomp, list):
        cube_der = np.zeros_like(cube_out)
        frame = []
        for nn, npc in enumerate(ncomp):
            cube_der[nn] = cube_derotate(cube_out[nn], angle_list, nproc=nproc,
                                         imlib=imlib,
                                         interpolation=interpolation,
                                         **rot_options)
            frame.append(cube_collapse(cube_der[nn], mode=collapse, w=weights))
    else:
        # Cube is derotated according to the parallactic angle and collapsed
        cube_der = cube_derotate(
            cube_out,
            angle_list,
            nproc=nproc,
            imlib=imlib,
            interpolation=interpolation,
            **rot_options,
        )
        frame = cube_collapse(cube_der, mode=collapse, w=weights)

    if verbose:
        print("Done derotating and combining.")
        timing(start_time)

    if full_output:
        return cube_out, cube_der, frame
    else:
        return frame
    
def _pca_adi_rdi_corr(
    cube,
    angle_list,
    epoch_indices,
    radius_int=0,
    fwhm=4,
    asize=2,
    n_segments=1,
    delta_rot=1,
    ncomp=1,
    svd_mode="lapack",
    nproc=None,
    step_corr = 1,
    max_frames_lib = None,
    ADI_Lib = None,
    RDI_Lib = None,
    tol=1e-1,
    scaling=None,
    imlib="vip-fft",
    interpolation="lanczos4",
    collapse="median",
    full_output=False,
    verbose=1,
    cube_ref=None,
    Mask_Corr = None,
    theta_init=0,
    weights=None,
    cube_sig=None,
    left_eigv=False,
    **rot_options,
):
    """PCA exploiting angular variability (ADI fashion)."""
    array = cube[epoch_indices[0]:epoch_indices[1]:1, :, :]
    if array.ndim != 3:
        raise TypeError("Input array is not a cube or 3d array")
    if cube.shape[0] != angle_list.shape[0]:
        raise TypeError("Input vector or parallactic angles has wrong length")

    n, y, x = array.shape
    
    cube_adi_ref = cube
    
    angle_list = check_pa_vector(angle_list)
    angle_list_adiref = angle_list
    
    angle_list = angle_list[epoch_indices[0]:epoch_indices[1]:1]
    
    n_annuli = int((y / 2 - radius_int) / asize)

    if isinstance(delta_rot, tuple):
        delta_rot = np.linspace(delta_rot[0], delta_rot[1], num=n_annuli)
    elif np.isscalar(delta_rot):
        delta_rot = [delta_rot] * n_annuli
    elif isinstance(delta_rot, list) and len(delta_rot) != n_annuli:
        raise ValueError("Delta_rot must have same length as the number of annuli")

    if isinstance(n_segments, int):
        n_segments = [n_segments for _ in range(n_annuli)]
    elif n_segments == "auto":
        n_segments = list()
        n_segments.append(2)  # for first annulus
        n_segments.append(3)  # for second annulus
        ld = 2 * np.tan(360 / 4 / 2) * asize
        for i in range(2, n_annuli):  # rest of annuli
            radius = i * asize
            ang = np.rad2deg(2 * np.arctan(ld / (2 * radius)))
            n_segments.append(int(np.ceil(360 / ang)))

    if verbose:
        msg = "N annuli = {}, FWHM = {:.3f}"
        print(msg.format(n_annuli, fwhm))
        print("PCA per annulus (or annular sectors):")

    if nproc is None:  # Hyper-threading "duplicates" the cores -> cpu_count/2
        nproc = cpu_count() // 2

    # The annuli are built, and the corresponding PA thresholds for frame
    # rejection are calculated (at the center of the annulus)
    cube_out = np.zeros_like(array)
    if isinstance(ncomp, list):
        nncomp = len(ncomp)
        cube_out = np.zeros([nncomp, array.shape[0], array.shape[1],
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

        n_segments_ann = n_segments[ann]
        res_ann_par = _define_annuli(
            angle_list,
            ann,
            n_annuli,
            fwhm,
            radius_int,
            asize,
            delta_rot[ann],
            n_segments_ann,
            verbose,
            True,
        )
        pa_thr, inner_radius, ann_center = res_ann_par
        indices = get_annulus_segments(
            array[0], inner_radius, asize, n_segments_ann, theta_init
        )

        if left_eigv:
            indices_out = get_annulus_segments(array[0], inner_radius, asize,
                                               n_segments_ann, theta_init,
                                               out=True)

        # Library matrix is created for each segment and scaled if needed
        for j in range(n_segments_ann):
            yy = indices[j][0]
            xx = indices[j][1]
            matrix_segm = array[:, yy, xx]  # shape [nframes x npx_segment]
            matrix_segm = matrix_scaling(matrix_segm, scaling)
            if cube_ref is not None:
                matrix_segm_ref = cube_ref[:, yy, xx]
                matrix_segm_ref = matrix_scaling(matrix_segm_ref, scaling)
            else:
                matrix_segm_ref = None
            if cube_sig is not None:
                matrix_sig_segm = cube_sig[:, yy, xx]
            else:
                matrix_sig_segm = None
                
            N_It = int(array.shape[0]/step_corr)

            if not left_eigv:
                res = pool_map(
                    nproc,
                    do_pca_patch_corr,
                    array,
                    epoch_indices,
                    xx,
                    yy,
                    iterable(range(N_It)),
                    step_corr,
                    angle_list,
                    fwhm,
                    pa_thr,
                    ann_center,
                    inner_radius,
                    asize,
                    svd_mode,
                    ncompann,
                    max_frames_lib,
                    ADI_Lib,
                    RDI_Lib,
                    scaling,
                    tol,
                    Mask_Corr,
                    cube_ref,
                    cube_adi_ref,
                    angle_list_adiref,
                    cube_sig,
                )

                if isinstance(ncomp, list):
                    nncomp = len(ncomp)
                    residuals = []
                    for nn in range(nncomp):
                        tmp = np.concatenate(tuple(res[i][0][nn] for i in range(0, N_It)))
                        residuals.append(tmp)
                    residuals = np.array(residuals)
                else:
                    residuals = np.concatenate(tuple(res[i][0] for i in range(0, N_It)))
                    ncomps = np.array(tuple(res[i][1] for i in range(0, N_It)))
                    nfrslib = np.array(tuple(res[i][2] for i in range(0, N_It)))
            else:
                yy_out = indices_out[j][0]
                xx_out = indices_out[j][1]
                matrix_out_segm = array[
                    :, yy_out, xx_out
                ]  # shape [nframes x npx_out_segment]
                matrix_out_segm = matrix_scaling(matrix_out_segm, scaling)
                if isinstance(ncomp, list):
                    npc = max(ncomp)
                else:
                    npc = ncomp
                V = get_eigenvectors(npc, matrix_out_segm, svd_mode,
                                     noise_error=tol, left_eigv=True)

                if isinstance(ncomp, list):
                    residuals = []
                    for nn, npc_tmp in enumerate(ncomp):
                        transformed = np.dot(V[:npc_tmp], matrix_segm.T)
                        reconstructed = np.dot(transformed.T, V[:npc_tmp])
                        residuals.append(matrix_segm - reconstructed)
                else:
                    transformed = np.dot(V, matrix_segm.T)
                    reconstructed = np.dot(transformed.T, V)
                    residuals = matrix_segm - reconstructed
                    nfrslib = matrix_out_segm.shape[0]

            if isinstance(ncomp, list):
                for nn, npc in enumerate(ncomp):
                    for i in range(0, cube_out.shape[1], 1):
                        cube_out[nn, i, yy, xx] = residuals[nn, i]
            else:
                for i in range(0, cube_out.shape[0], 1):
                    cube_out[i][yy, xx] = residuals[i]

            # number of frames in library printed for each annular quadrant
            # number of PCs printed for each annular quadrant
            if verbose == 2 and not isinstance(ncomp, list):
                descriptive_stats(nfrslib, verbose=verbose, label="\tLIBsize: ")
                descriptive_stats(ncomps, verbose=verbose, label="\tNum PCs: ")

        if verbose == 1:
            print("Done PCA with {} for current annulus".format(svd_mode))
            timing(start_time)

    if isinstance(ncomp, list):
        cube_der = np.zeros_like(cube_out)
        frame = []
        for nn, npc in enumerate(ncomp):
            cube_der[nn] = cube_derotate(cube_out[nn], angle_list, nproc=nproc,
                                         imlib=imlib,
                                         interpolation=interpolation,
                                         **rot_options)
            frame.append(cube_collapse(cube_der[nn], mode=collapse, w=weights))
    else:
        # Cube is derotated according to the parallactic angle and collapsed
        cube_der = cube_derotate(
            cube_out,
            angle_list,
            nproc=nproc,
            imlib=imlib,
            interpolation=interpolation,
            **rot_options,
        )
        frame = cube_collapse(cube_der, mode=collapse, w=weights)

    if verbose:
        print("Done derotating and combining.")
        timing(start_time)

    if full_output:
        return cube_out, cube_der, frame
    else:
        return frame
    

def do_pca_patch(
    matrix,
    frame,
    angle_list,
    fwhm,
    pa_threshold,
    ann_center,
    svd_mode,
    ncomp,
    min_frames_lib,
    max_frames_lib,
    tol,
    matrix_ref,
    matrix_sig_segm,
):
    """Do the SVD/PCA for each frame patch (small matrix).

    For each frame, frames to be rejected from the PCA library are found
    depending on the criterion in field rotation. The library is also truncated
    on the other end (frames too far in time, which have rotated more) which are
    more decorrelated, to keep the computational cost lower. This truncation is
    done on the annuli beyong 10*FWHM radius and the goal is to keep
    min(num_frames/2, 200) in the library.

    """
    if pa_threshold != 0:
        # if ann_center > fwhm*10:
        indices_left = _find_indices_adi2(angle_list, frame, pa_threshold,
                                         truncate=True,
                                         max_frames=max_frames_lib)
        msg = "Too few frames left in the PCA library. "
        msg += "Accepted indices length ({:.0f}) less than {:.0f}. "
        msg += "Try decreasing either delta_rot or min_frames_lib."
        try:
            if matrix_sig_segm is not None:
                data_ref = matrix[indices_left] - matrix_sig_segm[indices_left]
            else:
                data_ref = matrix[indices_left]
        except IndexError:
            if matrix_ref is None:
                raise RuntimeError(msg.format(0, min_frames_lib))
            data_ref = None

        if data_ref.shape[0] < min_frames_lib and matrix_ref is None:
            raise RuntimeError(msg.format(len(indices_left), min_frames_lib))
    else:
        if matrix_sig_segm is not None:
            data_ref = matrix - matrix_sig_segm
        else:
            data_ref = matrix

    if matrix_ref is not None:
        # Stacking the ref and the target ref (pa thresh) libraries
        if data_ref is not None:
            data_ref = np.vstack((matrix_ref, data_ref))
        else:
            data_ref = matrix_ref

    curr_frame = matrix[frame]  # current frame
    if matrix_sig_segm is not None:
        curr_frame_emp = matrix[frame] - matrix_sig_segm[frame]
    else:
        curr_frame_emp = curr_frame
    if isinstance(ncomp, list):
        npc = max(ncomp)
    else:
        npc = ncomp
    V = get_eigenvectors(npc, data_ref, svd_mode, noise_error=tol)

    if isinstance(ncomp, list):
        residuals = []
        for nn, npc_tmp in enumerate(ncomp):
            transformed = np.dot(curr_frame_emp, V[:npc_tmp].T)
            reconstructed = np.dot(transformed.T, V[:npc_tmp])
            residuals.append(curr_frame - reconstructed)
    else:
        transformed = np.dot(curr_frame_emp, V.T)
        reconstructed = np.dot(transformed.T, V)
        residuals = curr_frame - reconstructed

    return residuals, V.shape[0], data_ref.shape[0]


def do_pca_patch_corr(
    matrix,
    epoch_indices,
    xx,
    yy,
    batch,
    size_batch,
    angle_list,
    fwhm,
    pa_threshold,
    ann_center,
    radius_int,
    asize,
    svd_mode,
    ncomp,
    max_frames_lib,
    ADI_Lib,
    RDI_Lib,
    scaling,
    tol,
    mask_corr,
    matrix_rdi,
    matrix_adi,
    angle_list_adiref,
    matrix_sig,
):
    """Does the SVD/PCA for each frame patch (small matrix). For each frame we
    find the frames to be rejected depending on the amount of rotation. The
    library is also truncated on the other end (frames too far or which have
    rotated more) which are more decorrelated to keep the computational cost
    lower. This truncation is done on the annuli after 10*FWHM and the goal is
    to keep min(num_frames/2, 200) in the library.
    """
    Mask = np.zeros_like(matrix[0, :, :])
    Mask[xx, yy] = 1
    #SELECT ADI AND RDI IMAGES NOW WITH CORRELATION FUNCTION
    matrix_segm = matrix[:, yy, xx]  # shape [nframes x npx_segment]
    matrix_segm = matrix_scaling(matrix_segm, scaling)
    
    if matrix_sig is not None:
        matrix_sig_segm = matrix_sig[:, yy, xx]
    else:
        matrix_sig_segm = None
        
    
    if mask_corr is not None:
        matrix_corr = matrix * mask_corr
        matrix_adi_corr = matrix_adi * mask_corr
        matrix_rdi_corr = matrix_rdi * mask_corr
    else:
        matrix_corr = matrix
        matrix_adi_corr = matrix_adi
        matrix_rdi_corr = matrix_rdi
    
    if batch is not None:
        n_adi = matrix_adi.shape[0]
        indices_batch = batch * size_batch + np.array([i for i in range(0, size_batch)])
        
        if pa_threshold != 0:
            pa_range = np.max(angle_list[indices_batch[0]:indices_batch[-1]+1])-np.min(
                                     angle_list[indices_batch[0]:indices_batch[-1]+1]) 
            pa_limit = pa_range/2 + pa_threshold
            
            index = int((indices_batch[-1]+indices_batch[0])/2)
            truncate = False
            if max_frames_lib is not None:
                truncate = True
            indices_left = _find_indices_adi2(angle_list_adiref, index, 
                        pa_limit, truncate=truncate, max_frames=max_frames_lib)
            matrix_adi = matrix_adi[indices_left, : ,:]
            n_adi = matrix_adi.shape[0]
            #if n_adi < ADI_Lib:
            #    msg = "Pa_threshold too high. Not enough frames ({}) left in the library".format(n_adi)
            #    raise TypeError(msg)
        
        frame_ref = np.median(matrix_corr
                [indices_batch[0]:indices_batch[-1]+1:1, :, :], axis = 0)
        
        if (ADI_Lib is None) or (n_adi <= ADI_Lib and n_adi > 0):
            indices_adi = [list(np.arange(0, n_adi, 1))]
        elif n_adi != 0:
            percentile_adi = 100*(n_adi - ADI_Lib)/n_adi
            if mask_corr is None:
                indices_adi = cube_detect_badfr_correlation(matrix_adi, frame_ref,
                        percentile=percentile_adi, mode='annulus', inradius=radius_int,
                        width=asize, plot=False, verbose=False, 
                        crop_size=(radius_int+asize)*2)
            else:
                indices_adi = cube_detect_badfr_correlation(matrix_adi_corr, frame_ref,
                        percentile=percentile_adi, plot=False, verbose=False, 
                        crop_size=frame_ref.shape[0]-2)
        else:
            indices_adi = [[]]
        
        matrix_adi_ref = matrix_adi[:, yy, xx]
        matrix_adi_ref = matrix_adi_ref[indices_adi[0]]
        
        
        if matrix_rdi is not None:
            n_rdi = matrix_rdi.shape[0]
            if (RDI_Lib is None) or (RDI_Lib >= n_rdi):
                matrix_rdi_ref = matrix_rdi[:, yy, xx]
                data_ref = np.concatenate((matrix_adi_ref, matrix_rdi_ref))
            else:
                percentile_rdi = 100*(n_rdi - RDI_Lib)/n_rdi
                if mask_corr is None:
                    indices_rdi = cube_detect_badfr_correlation(matrix_rdi, frame_ref,
                        percentile=percentile_rdi, mode='annulus', inradius=radius_int,
                        width=asize, plot=False, verbose=False, 
                        crop_size=(radius_int+asize)*2)
                else:
                    indices_rdi = cube_detect_badfr_correlation(matrix_rdi_corr, 
                        frame_ref, percentile=percentile_rdi, plot=False, verbose=False, 
                        crop_size=frame_ref.shape[0]-2)
                matrix_rdi_ref = matrix_rdi[:, yy, xx]
                matrix_rdi_ref = matrix_rdi_ref[indices_rdi[0]]
                data_ref = np.concatenate((matrix_adi_ref, matrix_rdi_ref))
        else:
            data_ref = matrix_adi_ref
        
        data_ref = matrix_scaling(data_ref, scaling)
        
        if isinstance(ncomp, list):
            npc = max(ncomp)
        else:
            npc = ncomp
        V = get_eigenvectors(npc, data_ref, svd_mode, noise_error=tol)
        
        curr_frames = matrix_segm[indices_batch]  # current batch
        if matrix_sig_segm is not None:
            curr_frame_emp = matrix_segm[indices_batch] - matrix_sig_segm[indices_batch]
        else:
            curr_frame_emp = curr_frames
        
        
        if isinstance(ncomp, list):
            residuals = []
            for nn, npc_tmp in enumerate(ncomp):
                transformed = np.dot(V[:npc_tmp], curr_frame_emp.T)
                reconstructed = np.dot(transformed.T, V[:npc_tmp])
                residuals.append(curr_frames - reconstructed)
        else:
            transformed = np.dot(V, curr_frame_emp.T)
            reconstructed = np.dot(transformed.T, V)
            residuals = curr_frames - reconstructed
            
        return residuals, V.shape[0], data_ref.shape[0]
