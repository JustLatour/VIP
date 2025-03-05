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


def construct_rfrr_mask(input_mask, yy, xx, radius_mask, nbr_pixels):
    
    y,x = input_mask.shape
    coord_ann = list(zip(yy,xx))
    
    mask_array = np.ones((nbr_pixels, nbr_pixels))
    
    if radius_mask > 0:
        aperture = CircularAperture(np.array((xx, yy)).T, r=radius_mask)
        yy_m,xx_m = pxs_coord((y,x), aperture)
    else:
        yy_m = np.array([])
        xx_m = np.array([])
    
    for ap in range(0, nbr_pixels):
        mask_copy = np.copy(input_mask)
        if len(yy_m) > 0:
            mask_copy[yy_m[ap], xx_m[ap]] += 1
        yy_mask,xx_mask = np.where(mask_copy == 2)
        coord_ap = set(zip(yy_mask, xx_mask))
        
        indices = []
        for i, coord in enumerate(coord_ann):
            if coord in coord_ap:
                indices.append(i)
        indices = np.array(indices, dtype = int)
        
        mask_array[indices,ap] = 0
        
    mask_array = torch.tensor(mask_array, dtype = torch.float32)
    
    return mask_array



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
            nproc = 1, imlib = "vip-fft", interpolation = "lanczos4", 
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
        if device == 'cpu':
            print("'cuda' not available. Running on cpu instead.")
    
    if verbose:
        start = time.time()
    
    n,y,x = cube.shape
    
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
        yy = yy.flatten()
        xx = xx.flatten()
        
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
    
    mask_array = construct_rfrr_mask(annulus_mask, yy, xx, radius_mask * fwhm, nbr_pixels).to(device)
         
    matrix.requires_grad_(True)
    

    optimizer = torch.optim.LBFGS([matrix], lr=lr, max_iter=max_iter, history_size=history_size)

    #sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    #psf_model = gaussian_kernel(nbr_pixels, sigma)
    #psf_model = psf_model.unsqueeze(0)

    if not save_memory:
        grid_size = torch.tensor(cube).unsqueeze(1).size()
        all_grids = []
        radians = -angle_list * torch.pi / 180.0
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
                #this_col_image = torch.tensor(np.zeros((y, x)), dtype=torch.float32)
                #this_matrix = torch.tensor(np.zeros((nbr_pixels, nbr_pixels)), dtype=torch.float32)
                #for p in range(nbr_pixels):
                #    this_col = this_matrix_m[:,p]
                #    this_col_image[yy,xx] = this_col
                #    this_conv = masked_gaussian_convolution(this_col_image, mask_annular, kernel, kernel_size)
                #    this_matrix[:,p] = this_conv[yy,xx]
                this_matrix_cube = torch.zeros((nbr_pixels, y, x), device = device)
                this_matrix_cube[:,yy,xx] = this_matrix_m
                this_matrix_conv = F.conv2d(this_matrix_cube.unsqueeze(1),
                                       psf_model, padding = 'same').view(nbr_pixels,y,x)
                this_matrix = this_matrix_conv[:,yy,xx]
            else:
                this_matrix = this_matrix_m
            
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
            L2 = L2_penalty*torch.sum(matrix**2)
            if var:
                objective = torch.sum(torch.var(output_data_, axis=0))*n + L2
            else:
                objective = torch.mean(torch.std(output_data_, axis=0))*n + L2
            loss = objective
            
            # Backward pass
            loss.backward()
            
            # Apply gradient mask
            with torch.no_grad():
                matrix.grad *= mask_array
                
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
        
        
    output_data = (input_data - torch.matmul(input_data, matrix))*std
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