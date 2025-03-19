# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:15:35 2025

@author: JustinLat
"""

import numpy as np
from scipy.ndimage import shift
from multiprocessing import cpu_count
from typing import Tuple, Union, List
from dataclasses import dataclass
from enum import Enum
from .svd import svd_wrapper, SVDecomposer
from .utils_pca import pca_incremental, pca_grid
from ..config import (timing, time_ini, check_enough_memory, Progressbar,
                      check_array)
from ..config.paramenum import (
    SvdMode,
    Adimsdi,
    Interpolation,
    Imlib,
    Collapse,
    ALGO_KEY,
)
from ..config.utils_conf import pool_map, iterable
from ..config.utils_param import setup_parameters, separate_kwargs_dict
from ..preproc.derotation import _find_indices_adi, _compute_pa_thresh
from ..preproc import cube_rescaling_wavelengths as scwave
from ..preproc import (
    cube_derotate,
    cube_collapse,
    cube_subtract_sky_pca,
    check_pa_vector,
    check_scal_vector,
    cube_crop_frames,
    cube_detect_badfr_correlation
)
from ..stats import descriptive_stats
from ..var import (
    frame_center,
    dist,
    prepare_matrix,
    reshape_matrix,
    cube_filter_lowpass,
    mask_circle,
    get_annulus_segments, 
    frame_filter_lowpass
)

import time
import math
import torch
import torch.optim.lr_scheduler as lr_scheduler
import torch.optim as optim
from photutils.aperture import aperture_photometry, CircularAperture
import torch.nn.functional as F

from hciplot import plot_frames
import os

from fours.models.psf_subtraction import FourS
from fours.utils.pca import pca_psf_subtraction_gpu
from fours.utils.data_handling import save_as_fits
from fours.utils.data_handling import read_fours_root_dir
from pathlib import Path
import shutil
from vip_hci.fits import open_fits



def FourS_wrapper(cube, angle_list, psfn, fwhm, work_dir,lambda_reg = 10, rrmask = 1.5, 
                  num_epochs = 10, verbose = True, device = 0):
    these_angs = np.deg2rad(angle_list)

    s4_model = FourS(
        science_cube=cube,
        adi_angles=these_angs,
        psf_template=psfn,
        device=device,
        work_dir=work_dir,
        verbose=verbose,
        rotation_grid_subsample=1,
        noise_model_lambda=lambda_reg,
        psf_fwhm=fwhm,
        right_reason_mask_factor=rrmask)

    s4_model.fit_noise_model(
        num_epochs=num_epochs,
        training_name="Im-" + str(lambda_reg),
        logging_interval=1)

    mean, median, res, res_ = s4_model.compute_residuals()

    iterations = []
    folder_name = work_dir + '/residuals/' + os.listdir(work_dir + '/residuals')[0]
    image_names = os.listdir(folder_name)
    for f in image_names:
        if 'Mean' in f:
            continue
        it = open_fits(folder_name + '/' + f, verbose = False)
        iterations.append(it)
    iterations = np.array(iterations)

    shutil.rmtree(work_dir + '/residuals', ignore_errors=True)
    shutil.rmtree(work_dir + '/tensorboard', ignore_errors=True)
    shutil.rmtree(work_dir + '/models', ignore_errors=True)

    return np.array(res.squeeze(1)), np.array(res_.squeeze(1)), np.array(median), iterations




def limit_cpu_cores(core_ids):
    """Limit process to specific CPU cores."""
    original_affinity = os.sched_getaffinity(0)  # Save original core set
    os.sched_setaffinity(0, core_ids)            # Restrict to desired cores
    return original_affinity  

def restore_cpu_cores(original_affinity):
    """Restore process to original CPU core affinity."""
    os.sched_setaffinity(0, original_affinity)

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



def construct_round_rfrr_template(radius,psf_template_in):

    if radius == 0:
        template_mask = np.zeros_like(psf_template_in)
    else:
        image_center = int(psf_template_in.shape[-1] / 2)

        aperture = CircularAperture(positions=(image_center, image_center),
                                    r=radius)
        template_mask = aperture.to_mask().to_image(psf_template_in.shape)
    template = psf_template_in * template_mask
    
    template /= np.max(template)

    return template, template_mask


def construct_rfrr_mask2(cut_off_radius,
                        psf_template_in,
                        mask_in,
                        nbr_pixels):

    # 1.) Create the template
    template, template_mask = construct_round_rfrr_template(
        cut_off_radius,
        psf_template_in)
    
    side = mask_in.shape[0]
    regularization_mask = np.zeros((nbr_pixels, nbr_pixels))
    opp_mask = np.zeros((nbr_pixels, nbr_pixels))
    
    if template_mask.shape[0] < side:
        pad_size = int((side - template_mask.shape[0]) / 2)
        padded_template = np.pad(template_mask, pad_size)
    else:
        padded_template = template_mask
        
    center = side // 2
    pad = center
    
    if side % 2 == 1:
        addon = 1
    else:
        addon = 0

    yy,xx = np.where(mask_in == 1)

    k = 0
    for i in range(side):
        for j in range(side):
            if mask_in[i,j] != 1:
                continue
            this_image = np.zeros((side+2*pad, side+2*pad))
            this_opp = np.zeros((side+2*pad, side+2*pad))
            this_image[pad+i-pad:pad+i+pad+addon,pad+j-pad:pad+j+pad+addon] = padded_template
            this_opp[pad+side-1-i-pad:pad+side-1-i+pad+addon,pad+side-1-j-pad:pad+side-1-j+pad+addon] = padded_template

            this_image = this_image[pad:-pad, pad:-pad]
            this_image = 1 - this_image
            this_image *= mask_in

            this_opp = this_opp[pad:-pad, pad:-pad]
            this_opp = 1 - this_opp
            this_opp *= mask_in

            regularization_mask[:,k]=this_image[yy,xx]
            opp_mask[:,k]=this_opp[yy,xx]
        
            k += 1
            
    
    regularization_mask = torch.tensor(regularization_mask, dtype = torch.float32)
    opp_mask = torch.tensor(opp_mask, dtype = torch.float32)
    
    return regularization_mask, opp_mask



def construct_rfrr_mask(input_mask, yy, xx, radius_mask, nbr_pixels):
    
    y,x = input_mask.shape
    coord_ann = list(zip(yy,xx))
    
    mask_array = np.ones((nbr_pixels, nbr_pixels))
    opp_array = np.ones((nbr_pixels, nbr_pixels))
    
    if radius_mask > 0:
        aperture = CircularAperture(np.array((xx, yy)).T, r=radius_mask)
        yy_m,xx_m = pxs_coord((y,x), aperture)
    else:
        yy_m = np.array([])
        xx_m = np.array([])
    
    for ap in range(0, nbr_pixels):
        mask_copy = np.copy(input_mask)
        opp_copy = np.copy(input_mask)
        if len(yy_m) > 0:
            mask_copy[yy_m[ap], xx_m[ap]] += 1
            opp_copy[yy_m[nbr_pixels-ap-1], xx_m[nbr_pixels-ap-1]] += 1
        yy_mask,xx_mask = np.where(mask_copy == 2)
        yy_opp,xx_opp = np.where(opp_copy == 2)
        coord_ap = set(zip(yy_mask, xx_mask))
        coord_opp = set(zip(yy_opp, xx_opp))
        
        indices = [] 
        indices_opp = []
        for i, coord in enumerate(coord_ann):
            if coord in coord_ap:
                indices.append(i)
            if coord in coord_opp:
                indices_opp.append(i)
        indices = np.array(indices, dtype = int)
        indices_opp = np.array(indices_opp, dtype = int)
        
        mask_array[indices,ap] = 0
        opp_array[indices_opp,ap] = 0
        
    mask_array = torch.tensor(mask_array, dtype = torch.float32)
    opp_array = torch.tensor(opp_array, dtype = torch.float32)
    
    return mask_array, opp_array



def torch_rotate_image(image: torch.Tensor, degrees: float) -> torch.Tensor:
    """
    Rotate an image by an arbitrary number of degrees using PyTorch operations.

    Args:
        image (torch.Tensor): The image tensor of shape (C, H, W), where C is the number of channels.
        degrees (float): The angle in degrees to rotate the image.

    Returns:
        torch.Tensor: The rotated image tensor.
    """
    # Convert degrees to radians
    radians = degrees * torch.pi / 180.0
    
    image = image.unsqueeze(0)

    # Get image dimensions (C, H, W)
    _, height, width = image.shape

    # Create a rotation matrix
    cos_theta = torch.cos(radians)
    sin_theta = torch.sin(radians)
    rotation_matrix = torch.tensor([[cos_theta, -sin_theta, 0],
                                    [sin_theta,  cos_theta, 0]])

    # Compute the affine grid
    grid = F.affine_grid(rotation_matrix.unsqueeze(0), image.unsqueeze(0).size())

    # Apply the grid to rotate the image
    rotated_image = F.grid_sample(image.unsqueeze(0), grid, mode = 'bicubic', align_corners = True)

    # Remove the batch dimension and return the rotated image
    return rotated_image.squeeze(0).squeeze(0)
    

def torch_rotate_image_opt(image: torch.Tensor, degrees: float, grid) -> torch.Tensor:
    """
    Rotate an image by an arbitrary number of degrees using PyTorch operations.

    Args:
        image (torch.Tensor): The image tensor of shape (C, H, W), where C is the number of channels.
        degrees (float): The angle in degrees to rotate the image.

    Returns:
        torch.Tensor: The rotated image tensor.
    """
    # Convert degrees to radians
    radians = torch.tensor(degrees * torch.pi / 180.0)

    # Get image dimensions (C, H, W)
    height, width = image.shape
    
    image = image.unsqueeze(0).unsqueeze(0)

    # Create a rotation matrix
    cos_theta = torch.cos(radians)
    sin_theta = torch.sin(radians)
    rotation_matrix = torch.tensor([[cos_theta, -sin_theta, 0],
                                    [sin_theta,  cos_theta, 0]])

    # Apply the grid to rotate the image
    rotated_image = F.grid_sample(image, grid, mode = 'bicubic', align_corners = True)

    # Remove the batch dimension and return the rotated image
    return rotated_image.squeeze(0).squeeze(0)


def gaussian_kernel(size: int, sigma: float, device='cpu'):
    """Generates a 2D Gaussian kernel using PyTorch operations."""
    x_coord = torch.arange(size, device=device) - size // 2
    x_grid, y_grid = torch.meshgrid(x_coord, x_coord, indexing='ij')
    kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    return kernel

def masked_gaussian_convolution(image: torch.Tensor, 
                                mask: torch.Tensor, 
                                kernel,
                                kernel_size) -> torch.Tensor:
    """
    Applies a Gaussian convolution to the image only within the masked region.
    
    Parameters:
    image : 2D torch.Tensor
        Input image to be convolved (requires_grad enabled)
    mask : 2D boolean torch.Tensor
        Mask indicating the region to convolve (True where convolution is applied)
    fwhm : float
        Full width at half maximum of the Gaussian kernel
        
    Returns:
    torch.Tensor
        Convolved image with gradients preserved through the image parameter
    """
    # Handle None mask case
    if mask is None:
        mask = torch.ones_like(image, dtype=torch.bool)
    mask = mask.bool()

    # Prepare inputs with batch and channel dimensions
    image_bc = image.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    mask_bc = mask.unsqueeze(0).unsqueeze(0).float()  # [1, 1, H, W]

    # Compute numerator and denominator using convolution
    pad = kernel_size // 2
    padded_image = F.pad(image_bc * mask_bc, (pad, pad, pad, pad), mode='reflect')
    padded_mask = F.pad(mask_bc, (pad, pad, pad, pad), mode='reflect')

    numerator = F.conv2d(padded_image, kernel, padding=0)  # No padding since inputs are already padded
    denominator = F.conv2d(padded_mask, kernel, padding=0)

    # Avoid division by zero
    eps = 1e-6
    denominator = torch.where(denominator < eps, torch.tensor(eps, device=image.device), denominator)

    # Calculate normalized convolution and apply mask
    convolved_region = (numerator / denominator).squeeze()
    result = convolved_region * mask.float()

    return result


def torch_cube_derotate(array, angle_list, cyx, n):
    """Rotate a cube (3d array or image sequence) providing a vector or\
    corresponding angles.

    Serves for rotating an ADI sequence to a common north given a vector with
    the corresponding parallactic angles for each frame.

    Returns
    -------
    array_der : numpy ndarray
        Resulting cube with de-rotated frames.

    """
    array_der = torch.zeros_like(array)

    for i in range(n):
        array_der[i] = torch_rotate_image(array[i], -angle_list[i])
            
    return array_der


def torch_cube_derotate_batch(array, grids):
    """Rotate a cube (3d array or image sequence) providing a vector or\
    corresponding angles.

    Serves for rotating an ADI sequence to a common north given a vector with
    the corresponding parallactic angles for each frame.

    Returns
    -------
    array_der : numpy ndarray
        Resulting cube with de-rotated frames.

    """
    array_der = torch.zeros_like(array)

    array_der = F.grid_sample(array.unsqueeze(1), grids, mode = 'bicubic', align_corners = True).squeeze(1)
            
    return array_der


def annulus_4S(cube, angle_list, inner_radius, asize=4, fwhm = 4, psf_template = None,
            radius_mask = 0.75, L2_penalty = 0, iterations = 100, lr = 0.1,
            history_size = 10, max_iter = 20, limit = 0, verbose = False, 
            L2_exempt = False, psf_mask = True, std_norm = True,
            nproc = None, imlib = "vip-fft", interpolation = "lanczos4", 
            convolve = False, precision = 0.001, save_memory = False,
            var = False, device = None):
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if verbose:
            print(f"Using device: {device}")
    elif device != 'cpu' and device != 'cuda':
        raise ValueError("Device not recognized. Must be either 'cuda' or 'cpu'")
    elif device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if verbose:
            print(f'Using device: {device}')
        if device == 'cpu':
            print("'cuda' not available. Running on cpu instead.")
    
    if verbose:
        start = time.time()
    
    n,y,x = cube.shape
    
    if nproc is not None:
        if isinstance(nproc, list):
            original=limit_cpu_cores(nproc)
        else:
            raise ValueError("nproc must be None or a list")
        
    
    if asize is not None:
        if y % 2 == 0:
            new_size = (inner_radius + asize)*2 + 2
        else:
            new_size = (inner_radius + asize)*2 + 3
    
        if y > new_size:
            cube = cube_crop_frames(cube, new_size, verbose = False)
        
        yy,xx = get_annulus_segments(cube[0], inner_radius, asize, nsegm = 1, mode = 'ind')[0]
    else:
        yy,xx = np.meshgrid(np.arange(0,y), np.arange(0,y))
        yy = yy.T.flatten()
        xx = xx.T.flatten()
        
    n,y,x = cube.shape
        
        
    if psf_template is None:
        psf_size = int(fwhm*3)
        psf_model, _ = construct_round_rfrr_template(fwhm, psf_template_in=np.ones((psf_size,psf_size)))
    else:
        psf_model, _  = construct_round_rfrr_template(fwhm, psf_template_in=psf_template)
        
    psf_model = torch.tensor(psf_model, dtype=torch.float32, device = device).unsqueeze(0).unsqueeze(0)
    
    
    cyx = cube.shape[-1]/2

    mask_annular = np.zeros((y,x))
    mask_annular[yy,xx] = 1
    mask_annular = torch.tensor(mask_annular, dtype = torch.float32, device = device)

    inter_images = []
    
    annulus_mask = np.zeros_like(cube[0])
    annulus_mask[yy,xx] = 1
    
    nbr_pixels = len(yy)
    input_data = torch.tensor(cube[:,yy,xx], dtype = torch.float32, device = device)
    angle_list = torch.tensor(angle_list, dtype = torch.float32, device = device)
    mean = torch.mean(input_data, axis = 0)
    std = torch.std(input_data, axis = 0)
    input_data = (input_data-mean)/std
    
    matrix = torch.tensor(np.zeros((nbr_pixels, nbr_pixels)), dtype=torch.float32, device = device)
    
    if psf_mask:
        mask_array, opp_mask = construct_rfrr_mask2(radius_mask * fwhm,psf_template,annulus_mask,nbr_pixels)
    else:
        mask_array, opp_mask = construct_rfrr_mask(annulus_mask, yy, xx, radius_mask * fwhm, nbr_pixels)
    mask_array = mask_array.to(device)
    opp_mask = opp_mask.to(device)
         
    matrix.requires_grad_(True)
    

    optimizer = torch.optim.LBFGS([matrix], lr=lr, max_iter=max_iter, history_size=history_size)

    #sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    #psf_model = gaussian_kernel(nbr_pixels, sigma)
    #psf_model = psf_model.unsqueeze(0)

    if not save_memory:
        grid_size = torch.tensor(cube).unsqueeze(1).size()
        all_grids = []
        radians = - torch.deg2rad(angle_list)
        cos_theta = torch.cos(radians)
        sin_theta = torch.sin(radians)

        rotation_matrix = torch.stack(
            [cos_theta, -sin_theta,torch.zeros(n, device = device),
            sin_theta, cos_theta,torch.zeros(n, device = device)], dim=-1).view(-1, 2, 3)

        all_grids = F.affine_grid(rotation_matrix, grid_size, align_corners = True).to(device)

    if convolve:
        # Calculate Gaussian parameters
        sigma = fwhm / (2 * math.sqrt(2 * math.log(2)))
        kernel_size = 2 * int(3 * sigma) + 1  # Ensure odd kernel size

        # Generate Gaussian kernel
        kernel = gaussian_kernel(kernel_size, sigma, matrix.device)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
        
    #if convolve:
    #    mask_norm = F.conv2d(torch.tensor(annulus_mask).unsqueeze(0).unsqueeze(0), psf_model).view(y,x)

    prev = 0
    # Optimization loop
    for iteration in range(iterations):
        # Zero the gradients
        def closure():
            optimizer.zero_grad()
            
            this_matrix_m = matrix * mask_array
            
            if convolve:
                this_matrix_cube = torch.zeros((nbr_pixels, y, x), device = device)
                this_matrix_cube[:,yy,xx] = this_matrix_m
                #print('betas')
                #print(this_matrix_cube.shape)
                #print(this_matrix_cube)
                this_matrix_conv = F.conv2d(this_matrix_cube.unsqueeze(1),
                                       psf_model, padding = 'same').view(nbr_pixels,y,x)
                #.view does not put back data in the correct place. Need to transpose
                this_matrix_t = this_matrix_conv[:,yy,xx]
                this_matrix = this_matrix_t.T
            else:
                this_matrix = this_matrix_m
                
            #print('betas after conv')
            #print(this_matrix)
            #print('noise')
            #noise = torch.matmul(input_data, this_matrix.T)
            #print(noise.shape)
            #print(noise)
            
            output_data = input_data - torch.matmul(input_data, this_matrix)
            
            cube_data = torch.zeros((n,y,x), device = device)
            cube_data[:,yy,xx] = output_data

            if save_memory:
                cube_data_ = torch_cube_derotate(cube_data, angle_list, cyx, n)
            else:
                cube_data_ = torch_cube_derotate_batch(cube_data, all_grids)
                
            inter_images.append(np.median(cube_data_.detach().cpu().numpy(), axis = 0))
        
            output_data_ = torch.zeros((n, nbr_pixels), device = device)
            output_data_ = cube_data_[:,yy,xx]
            
            # Compute loss
            if L2_exempt:
                L2 = L2_penalty*torch.sum((matrix*opp_mask)**2)
            else:
                L2 = L2_penalty*torch.sum(matrix**2)
                
            if var:
                objective = torch.sum(torch.var(output_data_, axis=0))*n + L2
            else:
                objective = torch.mean(torch.std(output_data_, axis=0))*n + L2
            loss = objective
            
            #print(loss)
            
            # Backward pass
            loss.backward()
                
            #print('gradient')
            #print(matrix.grad.shape)
            #print(matrix.grad)
                
            return loss
    
        # L-BFGS optimization step
        loss = optimizer.step(closure)
    
        if verbose and (iteration + 1) % 2 == 0:
            print(f"Iteration {iteration + 1}: Objective = {loss.item()}")
        
        if iteration == 1:
            precision = precision * loss.item()
        
        if loss.item() < limit:
            break;

        if np.abs(loss.item() - prev) < precision:
            break
        prev = loss.item()
        
    
    if nproc is not None:
        nproc = None
        restore_cpu_cores(original)
        
    if std_norm:
        output_data = (input_data - torch.matmul(input_data, matrix))*std
    else:
        output_data = (input_data - torch.matmul(input_data, matrix))
    output_data = output_data.detach().cpu().numpy()
    
    cube_data = np.zeros((n,y,x))
    cube_data[:,yy,xx] = output_data

    angle_list = angle_list.detach().cpu().numpy()
    cube_data_ = cube_derotate(
                cube_data,
                angle_list,
                nproc=nproc,
                imlib=imlib,
                interpolation=interpolation, mask_val = 0, interp_zeros = True)
    
    
    mask_annular = np.zeros((y,x))
    mask_annular[yy,xx] = 1
    cube_data_ *= mask_annular

    result = np.median(cube_data_, axis = 0)
    result *= mask_annular
    
    if verbose:
        end = time.time()
        print('Algorithm ran for {} seconds'.format(end-start))
    
    return cube_data, cube_data_, result, loss.item(), matrix.detach().cpu().numpy(), np.array(inter_images)



def multi_cube_4S(big_cube, angle_list, inner_radius, asize=4, fwhm = 4, psf_template = None,
            radius_mask = 0.75, L2_penalty = 0, iterations = 100, lr = 0.1,
            history_size = 10, max_iter = 20, limit = 0, verbose = False, 
            L2_exempt = False, psf_mask = True, std_norm = True,
            nproc = None, imlib = "vip-fft", interpolation = "lanczos4", 
            convolve = False, precision = 0.001, save_memory = False,
            var = False, device = None):
    
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if verbose:
            print(f"Using device: {device}")
    elif device != 'cpu' and device != 'cuda':
        raise ValueError("Device not recognized. Must be either 'cuda' or 'cpu'")
    elif device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if verbose:
            print(f'Using device: {device}')
        if device == 'cpu':
            print("'cuda' not available. Running on cpu instead.")
    
    if verbose:
        start = time.time()
    
    nch = len(big_cube)
    n = np.zeros(nch, dtype = int)
    y = np.zeros(nch, dtype = int)
    x = np.zeros(nch, dtype = int)
    for c in range(nch):
        n[c],y[c],x[c] = big_cube[c].shape
    
    if nproc is not None:
        if isinstance(nproc, list):
            original=limit_cpu_cores(nproc)
        else:
            raise ValueError("nproc must be None or a list")
        
    
    cube = []
    if asize is not None:
        if y[-1] % 2 == 0:
            new_size = (inner_radius + asize)*2 + 2
        else:
            new_size = (inner_radius + asize)*2 + 3
            
        for c in range(nch):
            if y[-1] > new_size:
                cube.append(cube_crop_frames(big_cube[c], new_size, verbose = False))
            else:
                cube.append(big_cube[c])
        
        yy,xx = get_annulus_segments(cube[0][0], inner_radius, asize, nsegm = 1, mode = 'ind')[0]
    else:
        yy,xx = np.meshgrid(np.arange(0,y), np.arange(0,y))
        yy = yy.T.flatten()
        xx = xx.T.flatten()
        
        
    n = np.zeros(nch, dtype = int)
    y = np.zeros(nch, dtype = int)
    x = np.zeros(nch, dtype = int)
    total_im = np.zeros(nch+1, dtype = int)
    for c in range(nch):
        n[c],y[c],x[c] = cube[c].shape
        total_im[c+1:] = total_im[c+1:] + n[c]
    y = int(y[-1])
    x = int(x[-1])
        
    if np.isscalar(fwhm):
        fwhm = np.array([fwhm]*nch)
    if len(psf_template.shape) < 3:
        psf_template = [psf_template for c in range(nch)]
        
    psf_model = []
    psf_size = int(np.max(fwhm)*3)
    for c in range(nch):
        if psf_template is None:
            psf_model.append(construct_round_rfrr_template(fwhm[c], psf_template_in=np.ones((psf_size,psf_size)))[0])
        else:
            psf_model.append(construct_round_rfrr_template(fwhm[c], psf_template_in=psf_template[c])[0])
        
    psf_model = torch.tensor(np.array(psf_model), dtype=torch.float32, device = device).unsqueeze(0).unsqueeze(0)
    
    
    cyx = cube[-1].shape[-1]/2

    mask_annular = np.zeros((y,x))
    mask_annular[yy,xx] = 1
    mask_annular = torch.tensor(mask_annular, dtype = torch.float32, device = device)

    inter_images = []
    
    annulus_mask = np.zeros_like(cube[-1][0])
    annulus_mask[yy,xx] = 1
    
    if len(angle_list.shape) == 1:
        angle_list = np.array([angle_list for c in range(nch)])
    
    nbr_pixels = len(yy)
    input_data = []
    angle_lists = []
    mean = torch.zeros((nch,nbr_pixels), device = device, dtype = torch.float32)
    std = torch.zeros((nch,nbr_pixels), device = device, dtype = torch.float32)
    for c in range(nch):
        input_data.append(torch.tensor(cube[c][:,yy,xx], dtype = torch.float32, device = device))
        angle_lists.append(torch.tensor(angle_list[c], dtype = torch.float32, device = device))
        mean[c] = torch.mean(input_data[c], axis = 0)
        std[c] = torch.std(input_data[c], axis = 0)
        input_data[c] = (input_data[c]-mean[c])/std[c]
        input_data[c] = input_data[c].to(torch.float32)
    
    matrix = torch.tensor(np.zeros((nch,nbr_pixels, nbr_pixels)), dtype=torch.float32, device = device)
    
    mask_array = []
    opp_mask = []
    for c in range(nch):
        if psf_mask:
            this_mask_array, this_opp_mask = construct_rfrr_mask2(radius_mask * fwhm[c],psf_template,annulus_mask,nbr_pixels)
        else:
            this_mask_array, this_opp_mask = construct_rfrr_mask(annulus_mask, yy, xx, radius_mask * fwhm[c], nbr_pixels)

        mask_array.append(this_mask_array)
        opp_mask.append(this_opp_mask)
        
    mask_array = torch.tensor(np.array(mask_array), dtype = torch.float32, device = device)
    opp_mask = torch.tensor(np.array(opp_mask), dtype = torch.float32, device = device)
         
    matrix.requires_grad_(True)
    

    optimizer = torch.optim.LBFGS([matrix], lr=lr, max_iter=max_iter, history_size=history_size)

    #sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    #psf_model = gaussian_kernel(nbr_pixels, sigma)
    #psf_model = psf_model.unsqueeze(0)
    
    if not save_memory:
        all_grids = []
        for c in range(nch):
            grid_size = torch.tensor(cube[c]).unsqueeze(1).size()
            radians = - torch.deg2rad(angle_lists[c])
            cos_theta = torch.cos(radians)
            sin_theta = torch.sin(radians)

            rotation_matrix = torch.stack(
                [cos_theta, -sin_theta,torch.zeros(int(n[c]), device = device),
                 sin_theta, cos_theta,torch.zeros(int(n[c]), device = device)], dim=-1).view(-1, 2, 3)

            all_grids.append(F.affine_grid(rotation_matrix, grid_size, align_corners = True).to(device))

    kernel = []
    if convolve:
        # Calculate Gaussian parameters
        for c in range(nch):
            sigma = fwhm[c] / (2 * math.sqrt(2 * math.log(2)))
            kernel_size = 2 * int(3 * sigma) + 1  # Ensure odd kernel size

            # Generate Gaussian kernel
            kernel.append(gaussian_kernel(kernel_size, sigma, matrix.device))
            kernel[c] = kernel.unsqueeze(0).unsqueeze(0)  # Shape: [1, 1, H, W]
        
    #if convolve:
    #    mask_norm = F.conv2d(torch.tensor(annulus_mask).unsqueeze(0).unsqueeze(0), psf_model).view(y,x)

    prev = 0
    # Optimization loop
    torch.autograd.set_detect_anomaly(True)
    for iteration in range(iterations):
        # Zero the gradients
        def closure():
            optimizer.zero_grad()
            
            this_matrix_m = matrix * mask_array
            
            if convolve:
                this_matrix_cube = torch.zeros((nch,nbr_pixels, y, x), device = device)
                this_matrix_cube[:,:,yy,xx] = this_matrix_m
                
                this_matrix_conv = []
                this_matrix_t = []
                this_matrix = torch.zeros((nch,nbr_pixels,nbr_pixels))
                for c in range(nch):
                    this_matrix_conv.append(F.conv2d(this_matrix_cube[c].unsqueeze(1),
                                       psf_model, padding = 'same').view(nbr_pixels,y,x))
                    #.view does not put back data in the correct place. Need to transpose
                    this_matrix_t.append(this_matrix_conv[:,yy,xx])
                    this_matrix[c] = this_matrix_t[c].T
            else:
                this_matrix = this_matrix_m
                
                
            #print('betas after conv')
            #print(this_matrix)
            #print('noise')
            #noise = torch.matmul(input_data, this_matrix.T)
            #print(noise.shape)
            #print(noise)
            
            output_data = []
            #cube_data = torch.zeros((total_im[-1], y, x))
            cube_data_ = torch.zeros((total_im[-1], y, x), device = device)
            for c in range(nch):
                #input_data[c] = torch.tensor(input_data[c], dtype = torch.float32)
                output_data.append(input_data[c] - torch.matmul(input_data[c], this_matrix[c]))
            
                #this_cube_data = torch.zeros((n[c],y,x))
                #this_cube_data[:,yy,xx] = output_data[c]
                cube_data = torch.zeros((n[c], y, x), device = device)
                cube_data[:,yy,xx] = output_data[c]

                if save_memory:
                    cube_data_[total_im[c]:total_im[c+1]] = torch_cube_derotate(cube_data, angle_lists[c], cyx, n)
                else:
                    cube_data_[total_im[c]:total_im[c+1]] = torch_cube_derotate_batch(cube_data, all_grids[c])
                
            inter_images.append(np.median(cube_data_.detach().cpu().numpy(), axis = 0))
        
            output_data_ = torch.zeros((total_im[-1], nbr_pixels), device = device)
            output_data_ = cube_data_[:,yy,xx]
            
            # Compute loss
            if L2_exempt:
                L2 = L2_penalty*torch.sum((matrix*opp_mask)**2)
            else:
                L2 = L2_penalty*torch.sum(matrix**2)
                
            if var:
                objective = torch.sum(torch.var(output_data_, axis=0))*total_im[-1] + L2
            else:
                objective = torch.mean(torch.std(output_data_, axis=0))*total_im[-1] + L2
            loss = objective
            
            #print(loss)
            
            # Backward pass
            loss.backward()
                
            #print('gradient')
            #print(matrix.grad.shape)
            #print(matrix.grad)
                
            return loss
    
        # L-BFGS optimization step
        loss = optimizer.step(closure)
    
        if verbose and (iteration + 1) % 2 == 0:
            print(f"Iteration {iteration + 1}: Objective = {loss.item()}")
        
        if iteration == 1:
            precision = precision * loss.item()
        
        if loss.item() < limit:
            break;

        if np.abs(loss.item() - prev) < precision:
            break
        prev = loss.item()
        
    
    if nproc is not None:
        nproc = None
        restore_cpu_cores(original)
        
    cube_data = np.zeros((total_im[-1],y,x))
    cube_data_ = np.zeros((total_im[-1],y,x))
    for c in range(nch):
        angle_lists[c] = angle_lists[c].detach().cpu().numpy()
        input_data[c] = input_data[c].detach().cpu().numpy()
        
    std = std.detach().cpu().numpy()
    mean = mean.detach().cpu().numpy()
    
    output_data = []
    matrix = matrix.detach().cpu().numpy()
    for c in range(nch):
        if std_norm:
            output_data.append((input_data[c] - np.matmul(input_data[c], matrix[c]))*std[c])
        else:
            output_data.append((input_data[c] - np.matmul(input_data[c], matrix[c])))
    
        cube_data[total_im[c]:total_im[c+1],yy,xx] = output_data[c]

        cube_data_[total_im[c]:total_im[c+1]] = cube_derotate(
                cube_data[total_im[c]:total_im[c+1]],
                angle_lists[c],
                nproc=nproc,
                imlib=imlib,
                interpolation=interpolation, mask_val = 0, interp_zeros = True)
    
    
    mask_annular = np.zeros((y,x))
    mask_annular[yy,xx] = 1
    cube_data_ *= mask_annular

    result = np.median(cube_data_, axis = 0)
    result *= mask_annular
    
    if verbose:
        end = time.time()
        print('Algorithm ran for {} seconds'.format(end-start))
    
    return cube_data, cube_data_, result, loss.item(), matrix, np.array(inter_images)
