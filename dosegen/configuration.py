import json
import os
import ast
import glob
import zarr
import scipy
import shutil
import yaml
import argparse
import nibabel as nib
import numpy as np
import torch
import gc
import copy

from numcodecs import Blosc
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from train_autoencoder import AutoEncoder
from data_processing import get_data_loaders


def compute_downsample_parameters(input_size, num_layers):
    """
    Generalized to handle 1D, 2D, or 3D input sizes.

    Args:
        input_size: list of ints [D, H, W] or [H, W] or [W]
        num_layers: int, number of layers including the first one

    Returns:
        List of lists: [[[stride], [kernel], [padding]], ...] for each layer
    """
    ndim = len(input_size)
    current_size = list(input_size)
    parameters = []

    for i in range(num_layers):
        stride = [1] * ndim
        kernel = [3] * ndim
        padding = [1] * ndim

        if i == 0:
            # First layer: adjust based on dimension disparity
            for d in range(ndim):
                other_dims = [current_size[j] for j in range(ndim) if j != d]
                if current_size[d] <= 0.5 * max(other_dims, default=current_size[d]):
                    kernel[d] = 1
                    padding[d] = 0
        else:
            # Downsampling layers
            for d in range(ndim):
                other_dims = [current_size[j] for j in range(ndim) if j != d]
                if current_size[d] <= 0.5 * max(other_dims, default=current_size[d]):
                    stride[d] = 1
                    kernel[d] = 1
                    padding[d] = 0
                else:
                    stride[d] = 2
                    kernel[d] = 3
                    padding[d] = 1

            # Update size after downsampling
            for d in range(ndim):
                current_size[d] = (current_size[d] + 2 * padding[d] - kernel[d]) // stride[d] + 1

        parameters.append([stride, kernel, padding])

    return parameters


def compute_output_size(input_size, downsample_parameters):
    """
    Args:
        input_size: list of ints (e.g. [D, H, W] or [H, W])
        downsample_parameters: list of [[stride], [kernel], [padding]] per layer

    Returns:
        output_size: list of ints representing final size after all layers
    """
    output_size = list(input_size)

    for layer in downsample_parameters:
        stride, kernel, padding = layer
        for d in range(len(output_size)):
            output_size[d] = (
                (output_size[d] + 2 * padding[d] - kernel[d]) // stride[d]
            ) + 1

    return output_size


def create_autoencoder_dict(dataset_config, input_channels, spatial_dims):
    median_image_size = dataset_config['median_shape']
    max_image_size = dataset_config['max_shape']
    # For 2d, for each axis, use as size the closest multiple of 2, 3, 5 or 7 by powers of 2, to the corresponding size of max patch size
    valid_2d_sizes = [32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512]
    patch_size_2d = [min(valid_2d_sizes, key=lambda x: abs(x - size)) for size in max_image_size]
    # For 3d, for each axis, use as size the closest multiple of 2, 3, or 7 by 2, to the corresponding size of nnunet median patch size
    valid_3d_sizes = [32, 48, 56, 64, 96, 112, 128, 192, 224, 256, 384, 448, 512]
    patch_size_3d = [min(valid_3d_sizes, key=lambda x: abs(x - size)) for size in median_image_size]
    patch_size = patch_size_2d[1:] if spatial_dims == 2 else patch_size_3d

    base_autoencoder_channels = [64, 128, 256, 256] if spatial_dims == 2 else [32, 64, 128, 128]

    vae_dict = {'spatial_dims': spatial_dims,
                'in_channels': len(input_channels),
                'out_channels': len(input_channels),
                'latent_channels': 8,
                'num_res_blocks': 2,
                'with_encoder_nonlocal_attn': False,
                'with_decoder_nonlocal_attn': False,
                'use_flash_attention': False,
                'use_checkpointing': False,
                'use_convtranspose': False
               }

    # use maximum of 3 autoencoder downsampling layers
    # we want the latent dims to be less than 100 to be managable (say less than 96)
    if np.max(patch_size) <= 96:
        vae_n_layers = 1
    elif np.max(patch_size) <= 384:
        vae_n_layers = 2
    else:
        vae_n_layers = 3

    vae_dict['num_channels'] = base_autoencoder_channels[:vae_n_layers+1]
    vae_dict['attention_levels'] = [False] * (vae_n_layers+1)
    vae_dict['norm_num_groups'] = 16

    downsample_parameters = compute_downsample_parameters(patch_size, vae_n_layers + 1)
    vae_dict['downsample_parameters'] = downsample_parameters
    vae_dict['upsample_parameters'] = list(reversed(downsample_parameters))[:-1]
    return vae_dict


def create_ddpm_dict(dataset_config, spatial_dims):
    median_image_size = dataset_config['median_shape']
    max_image_size = dataset_config['max_shape']
    # For 2d, for each axis, use as size the closest multiple of 2, 3, 5 or 7 by powers of 2, to the corresponding size of max patch size
    valid_2d_sizes = [32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512]
    patch_size_2d = [min(valid_2d_sizes, key=lambda x: abs(x - size)) for size in max_image_size]
    # For 3d, for each axis, use as size the closest multiple of 2, 3, or 7 by 2, to the corresponding size of nnunet median patch size
    valid_3d_sizes = [32, 48, 56, 64, 96, 112, 128, 192, 224, 256, 384, 448, 512]
    patch_size_3d = [min(valid_3d_sizes, key=lambda x: abs(x - size)) for size in median_image_size]
    patch_size = patch_size_2d[1:] if spatial_dims == 2 else patch_size_3d

    ddpm_dict = {'spatial_dims': spatial_dims,
                 'in_channels': 8,
                 'out_channels': 8,
                 'num_res_blocks': 2,
                 'use_flash_attention': False,
                }

    # use maximum of 3 autoencoder downsampling layers
    # we want the latent dims to be less than 100 to be managable (say less than 96)
    if np.max(patch_size) <= 96:
        vae_n_layers = 1
    elif np.max(patch_size) <= 384:
        vae_n_layers = 2
    else:
        vae_n_layers = 3

    ddpm_dict['num_channels'] = [256, 512, 768]
    ddpm_dict['attention_levels'] = [False, True, True]
    ddpm_dict['num_head_channels'] = [0, 512, 768]

    vae_down_params = compute_downsample_parameters(patch_size, vae_n_layers + 1)
    latent_size = compute_output_size(patch_size, vae_down_params)
    ddpm_down_params = compute_downsample_parameters(latent_size, 3)

    ddpm_dict['strides'] = [item[0] for item in ddpm_down_params]
    ddpm_dict['kernel_sizes'] = [item[1] for item in ddpm_down_params]
    ddpm_dict['paddings'] = [item[2] for item in ddpm_down_params]

    return ddpm_dict


def create_config_dict(dataset_config, input_channels, n_epochs_multiplier, autoencoder_dict, ddpm_dict):
    median_image_size = dataset_config['median_shape']
    max_image_size = dataset_config['max_shape']
    # For 2d, for each axis, use as size the closest multiple of 2, 3, 5 or 7 by powers of 2, to the corresponding size of max patch size
    valid_2d_sizes = [32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 192, 224, 256, 320, 384, 448, 512]
    patch_size_2d = [min(valid_2d_sizes, key=lambda x: abs(x - size)) for size in max_image_size]
    # For 3d, for each axis, use as size the closest multiple of 2, 3, or 7 by 2, to the corresponding size of nnunet median patch size
    valid_3d_sizes = [32, 48, 56, 64, 96, 112, 128, 192, 224, 256, 384, 448, 512]
    patch_size_3d = [min(valid_3d_sizes, key=lambda x: abs(x - size)) for size in median_image_size]
    patch_size = patch_size_2d[1:] if autoencoder_dict['spatial_dims'] == 2 else patch_size_3d

    if autoencoder_dict['spatial_dims'] == 2:
        batch_size = 24
    else:
        batch_size = 2

    print(f"Patch size {autoencoder_dict['spatial_dims']}D: {patch_size}")

    ae_transformations = {
        "patch_size": patch_size,
        "scaling": False,
        "rotation": True,
        "gaussian_noise": False,
        "gaussian_blur": False,
        "low_resolution": False,
        "brightness": False,
        "contrast": False,
        "gamma": False,
        "mirror": True,
        "dummy_2d": False
    }
    # not using augmentations for ddpm training
    ddpm_transformations = {
        "patch_size": patch_size,
        "scaling": False,
        "rotation": False,
        "gaussian_noise": False,
        "gaussian_blur": False,
        "low_resolution": False,
        "brightness": False,
        "contrast": False,
        "gamma": False,
        "mirror": True,
        "dummy_2d": False
    }

    if autoencoder_dict['spatial_dims'] == 2:
        perceptual_params = {'spatial_dims': 2, 'network_type': "vgg"}
    else:
        perceptual_params = {'spatial_dims': 3, 'network_type': "vgg", 'is_fake_3d': True, 'fake_3d_ratio': 0.2}

    discriminator_params = {'spatial_dims': autoencoder_dict['spatial_dims'], 'in_channels': autoencoder_dict['in_channels'],
                            'out_channels': 1, 'num_channels': 64, 'num_layers_d': 3}

    # adjust the number of epochs based on the training model (2d/3d) and number of training data
    n_epochs = 300 if autoencoder_dict['spatial_dims'] == 3 else 200
    n_epochs = n_epochs * n_epochs_multiplier

    ae_batch_size = batch_size
    ddpm_batch_size = ae_batch_size * 2
    grad_accumulate_step = 1

    config = {
        'input_channels': input_channels,
        'ae_transformations': ae_transformations,
        'ddpm_transformations': ddpm_transformations,
        'ae_batch_size': ae_batch_size,
        'ddpm_batch_size': ddpm_batch_size,
        'n_epochs': n_epochs,
        'val_plot_interval': 10,
        'grad_clip_max_norm': 1,
        'grad_accumulate_step': grad_accumulate_step,
        'oversample_ratio': 0.33,
        'num_workers': 8,
        'lr_scheduler': "PolynomialLR",
        'lr_scheduler_params': {'total_iters': n_epochs, 'power': 0.9},
        'time_scheduler_params': {'num_train_timesteps': 1000, 'schedule': "scaled_linear_beta", 'beta_start': 0.0015,
                                  'beta_end': 0.0205, 'prediction_type': "epsilon"},
        'ae_learning_rate': 5e-5,
        'd_learning_rate': 5e-5,
        'autoencoder_warm_up_epochs': 5,
        'adv_weight': 0.01,
        'perc_weight': 0.5 if autoencoder_dict['spatial_dims'] == 2 else 0.125,
        'kl_weight': 1e-6 if autoencoder_dict['spatial_dims'] == 2 else 1e-7,
        'vae_params': autoencoder_dict,
        'perceptual_params': perceptual_params,
        'discriminator_params': discriminator_params,
        'ddpm_learning_rate': 2e-5,
        'ddpm_params': ddpm_dict
    }
    return config


def extract_spacing(path):
    img = nib.load(path)
    spacing = np.sqrt(np.sum(img.affine[:3, :3] ** 2, axis=0))  # voxel spacing from affine
    return spacing


def calculate_median_spacing(patient_dict):
    with ProcessPoolExecutor() as executor:
        image_spacings = list(executor.map(
            lambda p: extract_spacing(p['image']),
            patient_dict.values()
        ))

    return tuple(np.median(image_spacings, axis=0))


def is_anisotropic(spacing, threshold=3.0):
    return (np.max(spacing) / np.min(spacing)) > threshold


def resample_image_dose_label(image, dose, label, target_spacing):
    image_data = image.get_fdata()
    dose_data = dose.get_fdata()
    label_data = label.get_fdata()

    original_spacing = np.sqrt(np.sum(image.affine[:3, :3] ** 2, axis=0))
    zoom_factors = original_spacing / target_spacing
    anisotropic = is_anisotropic(original_spacing)

    log_lines = []
    if np.allclose(original_spacing, target_spacing):
        log_lines.append("    No resampling needed")
        return image_data, dose_data, label_data, log_lines

    log_lines.append("    Difference with target spacing. Resampling items...")
    log_lines.append(f"        Original spacing: {original_spacing} - Final spacing: {target_spacing}")

    lowres_axis = np.argmax(original_spacing)
    image_order = [3 if not anisotropic or i != lowres_axis else 0 for i in range(3)]
    label_order = [1 if not anisotropic or i != lowres_axis else 0 for i in range(3)]

    resampled_image = image_data
    resampled_dose = dose_data
    for axis in range(3):
        if zoom_factors[axis] != 1:
            zoom = [zoom_factors[axis] if i == axis else 1 for i in range(3)]
            resampled_image = scipy.ndimage.zoom(resampled_image, zoom=zoom, order=image_order[axis])
            resampled_dose = scipy.ndimage.zoom(resampled_dose, zoom=zoom, order=image_order[axis])

    unique_labels = np.unique(label_data)
    unique_labels = unique_labels[unique_labels != 0]  # exclude background
    one_hot = np.stack([label_data == cls for cls in unique_labels], axis=0)

    resampled_channels = []
    for c in range(one_hot.shape[0]):
        channel = one_hot[c].astype(np.float32)
        for axis in range(3):
            if zoom_factors[axis] != 1:
                zoom = [zoom_factors[axis] if i == axis else 1 for i in range(3)]
                channel = scipy.ndimage.zoom(channel, zoom=zoom, order=label_order[axis])
        resampled_channels.append(channel)

    argmax_output = np.argmax(np.stack(resampled_channels, axis=0), axis=0)
    resampled_label = np.zeros_like(argmax_output, dtype=np.uint8)
    for idx, cls in enumerate(unique_labels):
        resampled_label[argmax_output == idx] = cls

    return resampled_image, resampled_dose, resampled_label, log_lines


def normalize_zscore_then_minmax(image):
    normalized = np.zeros_like(image, dtype=np.float32)
    min_max_per_channel = []

    for c in range(image.shape[0]):
        channel_data = image[c]

        vmin = np.min(channel_data)
        vmax = np.max(channel_data)

        z_image = (channel_data - np.mean(channel_data)) / np.std(channel_data)
        z_min = np.min(z_image)
        z_max = np.max(z_image)
        normalized[c] = (z_image - z_min) / (z_max - z_min)

        min_max_per_channel.append((vmin, vmax))

    return normalized, min_max_per_channel


def get_resampled_shape_and_channel_min_max(patient_paths, median_spacing):
    img = nib.load(patient_paths['image'])
    dose = nib.load(patient_paths['dose'])
    label = nib.load(patient_paths['label'])

    resampled_image, resampled_dose, *_ = resample_image_dose_label(img, dose, label, target_spacing=median_spacing)

    if resampled_image.ndim == 3:
        resampled_image = np.expand_dims(resampled_image, axis=-1)

    resampled_image = np.transpose(resampled_image, (3, 2, 1, 0))
    _, img_min_max_per_channel = normalize_zscore_then_minmax(resampled_image)

    if resampled_dose.ndim == 3:
        resampled_dose = np.expand_dims(resampled_dose, axis=-1)

    resampled_dose = np.transpose(resampled_dose, (3, 2, 1, 0))
    _, dose_min_max_per_channel = normalize_zscore_then_minmax(resampled_dose)

    return resampled_image.shape, img_min_max_per_channel, dose_min_max_per_channel


def calculate_dataset_shapes_and_channel_min_max(patient_dict, median_spacing):
    # Prepare function with fixed spacing
    fn = partial(get_resampled_shape_and_channel_min_max, median_spacing=median_spacing)

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(fn, patient_dict))

    shapes, img_min_max_per_channel, dose_min_max_per_channel = zip(*results)
    median_shape = tuple(np.median(np.array(shapes), axis=0).astype(int))
    min_shape = tuple(np.min(np.array(shapes), axis=0).astype(int))
    max_shape = tuple(np.max(np.array(shapes), axis=0).astype(int))

    img_min_max_per_channel = np.array(img_min_max_per_channel)  # Shape: (num_images, num_channels, 2)
    img_global_channel_min = img_min_max_per_channel[..., 0].min(axis=0)
    img_global_channel_max = img_min_max_per_channel[..., 1].max(axis=0)

    dose_min_max_per_channel = np.array(dose_min_max_per_channel)  # Shape: (num_images, num_channels, 2)
    dose_global_channel_min = dose_min_max_per_channel[..., 0].min(axis=0)
    dose_global_channel_max = dose_min_max_per_channel[..., 1].max(axis=0)

    return (median_shape, min_shape, max_shape, img_global_channel_min.tolist(), img_global_channel_max.tolist(),
            dose_global_channel_min.tolist(), dose_global_channel_max.tolist())


def get_sampled_class_locations(label_array, samples_per_slice=50):
    class_locations = {}
    unique_labels = np.unique(label_array)

    for lbl in unique_labels:
        if lbl == 0:
            continue  # skip background

        coords = []
        for z in range(label_array.shape[0]):
            slice_mask = label_array[z] == lbl
            slice_coords = np.argwhere(slice_mask)

            if slice_coords.shape[0] == 0:
                continue  # no voxels for this label in this slice

            if slice_coords.shape[0] > samples_per_slice:
                indices = np.random.choice(slice_coords.shape[0], samples_per_slice, replace=False)
                sampled = slice_coords[indices]
            else:
                sampled = slice_coords

            # Add Z back as the first coordinate
            sampled = [(z, y, x) for y, x in sampled]
            coords.extend(sampled)

        class_locations[int(lbl)] = coords

    return class_locations


def process_patient(patient_id, patient_dict, save_path, median_spacing, median_shape):
    log_lines = [f"Processing {patient_id}..."]

    file_save_path = os.path.join(save_path, patient_id + '.zarr')

    image = nib.load(patient_dict[patient_id]['image'])
    dose = nib.load(patient_dict[patient_id]['dose'])
    label = nib.load(patient_dict[patient_id]['label'])

    resampled_image, resampled_dose, resampled_label, resample_log_lines = resample_image_dose_label(image, dose, label, median_spacing)
    log_lines.extend(resample_log_lines)

    if resampled_image.ndim == 3:
        resampled_image = np.expand_dims(resampled_image, axis=-1)
    resampled_image = np.transpose(resampled_image, (3, 2, 1, 0))

    if resampled_dose.ndim == 3:
        resampled_dose = np.expand_dims(resampled_dose, axis=-1)
    resampled_dose = np.transpose(resampled_dose, (3, 2, 1, 0))
    resampled_dose = np.squeeze(resampled_dose, axis=0)

    resampled_label = np.transpose(resampled_label, (2, 1, 0))

    normalized_image, img_min_max = normalize_zscore_then_minmax(resampled_image)
    normalized_dose, dose_min_max = normalize_zscore_then_minmax(resampled_dose)

    unique_labels = np.unique(resampled_label).tolist()
    class_locations = get_sampled_class_locations(resampled_label, samples_per_slice=50)

    properties = {'class_locations': class_locations, 'img_min_max': img_min_max, 'dose_min_max': dose_min_max}

    compressor = Blosc(cname='zstd', clevel=5, shuffle=Blosc.BITSHUFFLE)
    image_chunks = (1, 1) + tuple(median_shape[-2:])
    label_or_dose_chunks = (1,) + tuple(median_shape[-2:])
    z_file = zarr.open(file_save_path, mode='w')
    z_file.create_array(name='image', data=normalized_image.astype(np.float32), chunks=image_chunks, compressor=compressor, overwrite=True)
    z_file.create_array(name='dose', data=normalized_dose.astype(np.float32), chunks=label_or_dose_chunks, compressor=compressor, overwrite=True)
    z_file.create_array(name='label', data=resampled_label.astype(np.uint8), chunks=label_or_dose_chunks, compressor=compressor, overwrite=True)
    z_file.attrs['properties'] = properties

    log_lines.append(f"    Saved processed image, dose, label and properties to {file_save_path}")

    return {
        "patient_id": patient_id,
        "shape": normalized_image.shape,
        "labels": [item for item in unique_labels if item != 0],
        "log": "\n".join(log_lines)
    }


def process_patient_wrapper(args):
    return process_patient(*args)


def auto_select_hyperparams(dataset_id, model_fn, config, model_type='2d', init_batch_size=48, init_grad_accum=1):

    assert model_type in ['2d', '3d'], "model_type must be either '2d' or '3d'"

    batch_size = init_batch_size
    grad_accum = init_grad_accum

    min_batch_size = 6 if model_type == '2d' else 1

    preprocessed_dataset_path = glob.glob(os.getenv('medimgen_preprocessed') + f'/Task{dataset_id}*/')[0]
    dataset_folder_name = preprocessed_dataset_path.split('/')[-2]
    results_path = os.path.join(os.getenv('medimgen_results'), dataset_folder_name, model_type, 'autoencoder')

    def try_run(batch_size, grad_accum):
        try:
            # Free memory
            torch.cuda.empty_cache()
            gc.collect()

            # Update config with new batch size and grad accum
            test_config = copy.deepcopy(config)
            test_config['ae_batch_size'] = batch_size
            test_config['grad_accumulate_step'] = grad_accum
            test_config['n_epochs'] = 1

            test_config['progress_bar'] = False
            test_config['output_mode'] = 'verbose'
            test_config['results_path'] = results_path
            test_config['load_model_path'] = None

            # Rebuild model and data loaders
            model = model_fn(config=test_config, latent_space_type='vae', print_summary=False)
            transformations = test_config['ae_transformations']
            train_loader, val_loader = get_data_loaders(test_config, dataset_id, 'train-val-test', batch_size, model_type, transformations)

            # Try training for a short time (1 epoch)
            model.train(train_loader=train_loader, val_loader=val_loader)
            print(f"We will use batch size = {batch_size} and grad_accumulate_step = {grad_accum} while training in {model_type}.")
            if os.path.exists(os.path.join(os.getenv('medimgen_results'), dataset_folder_name)):
                shutil.rmtree(os.path.join(os.getenv('medimgen_results'), dataset_folder_name))
            return True

        except RuntimeError as e:
            if os.path.exists(os.path.join(os.getenv('medimgen_results'), dataset_folder_name)):
                shutil.rmtree(os.path.join(os.getenv('medimgen_results'), dataset_folder_name))
            if any([item in str(e) for item in ["CUDA out of memory", "Failed to run torchinfo"]]):
                print(f"[OOM] BatchSize: {batch_size}, GradAccumSteps: {grad_accum}")
                del model
                torch.cuda.empty_cache()
                gc.collect()
                return False
            else:
                raise e

    # Try initial setting
    if try_run(batch_size, grad_accum):
        return batch_size, grad_accum

    if model_type == '2d':
        grad_accum = 2
        while batch_size > min_batch_size:
            batch_size //= 2
            if try_run(batch_size, grad_accum):
                return batch_size, grad_accum

        if try_run(min_batch_size, grad_accum):
            return min_batch_size, grad_accum
        else:
            print(f"Warning! 2d model cannot fit even with batch_size = {batch_size} and grad_accumulate_step = {grad_accum}. You need a bigger GPU!")
            return batch_size, grad_accum

    elif model_type == '3d':
        batch_size //= 2
        grad_accum = 2
        if try_run(batch_size, grad_accum):
            return batch_size, grad_accum
        else:
            print(f"Warning! 3d model cannot fit even with batch_size = {batch_size} and grad_accumulate_step = {grad_accum}. You need a bigger GPU!")
            return batch_size, grad_accum


def main():
    parser = argparse.ArgumentParser(description="Preprocess dataset and create configuration file.")
    parser.add_argument("dataset_path", type=str, help="Path to dataset folder")

    args = parser.parse_args()
    dataset_path = args.dataset_path

    basename = os.path.basename(dataset_path)
    dataset_id = basename.split('_')[0][4:]
    # format the task number to 3 digits with leading zeros
    formatted_task_number = f"{int(dataset_id):03d}"
    # standardized folder name
    standardized_folder_name = f"Task{formatted_task_number}_" + "_".join(basename.split('_')[1:])
    dataset_save_path = os.path.join(os.getenv('dosegen_preprocessed'), standardized_folder_name)

    if os.path.exists(dataset_save_path):
        raise FileExistsError(f"Dataset {os.path.basename(dataset_path)} already exists.")

    # Given dataset folder must have a name in the form TaskXXX_DatasetName
    # The dataset folder should always contain an 'images' folder, a 'doses' folder and a 'labels' folder
    # All the files should be .nii.gz files

    os.makedirs(dataset_save_path, exist_ok=True)

    image_paths = glob.glob(os.path.join(dataset_path, 'images') + "/*.nii.gz")
    patient_ids = sorted([os.path.basename(path).replace('.nii.gz', '') for path in image_paths])
    patient_dict = {}
    for pid in patient_ids:
        img_path = os.path.join(dataset_path, 'images', f"{pid}.nii.gz")
        dose_path = os.path.join(dataset_path, 'dose', f"{pid}.nii.gz")
        label_path = os.path.join(dataset_path, 'labels', f"{pid}.nii.gz")
        patient_dict[pid] = {'image': img_path, 'dose': dose_path, 'label': label_path}

    print(f"\nNumber of patients: {len(patient_ids)}")
    print("\nCalculating median voxel spacing of the whole dataset...")
    median_spacing = calculate_median_spacing(patient_dict)
    print(
        "Calculating dataset min and max values, median, min, and max shape after cropping and resampling, and low quality images...")
    dataset_results = calculate_dataset_shapes_and_channel_min_max(patient_dict, median_spacing)
    median_shape, min_shape, max_shape, img_global_channel_min, img_global_channel_max, dose_global_channel_min, dose_global_channel_max = dataset_results
    print(f"\nMedian voxel spacing: {median_spacing}")
    print(f"Median Shape: {median_shape}")
    print(f"Min Shape: {min_shape}")
    print(f"Max Shape: {max_shape}")
    print(f"Image Min per channel: {img_global_channel_min}")
    print(f"Image Max per channel: {img_global_channel_max}")
    print(f"Dose Min per channel: {dose_global_channel_min}")
    print(f"Dose Max per channel: {dose_global_channel_max}")

    median_shape_w_channel = median_shape
    median_shape, min_shape, max_shape = median_shape[1:], min_shape[1:], max_shape[1:]

    results = []
    args_list = [(pid, patient_dict, dataset_save_path, median_spacing, median_shape)
                 for pid in patient_ids]

    with ProcessPoolExecutor() as executor:
        for result in executor.map(process_patient_wrapper, args_list):
            print(result["log"])
            results.append(result)

    all_labels = [lbl for r in results for lbl in r["labels"]]
    unique_labels = sorted(set(all_labels))
    n_patients = len(results)
    n_channels = median_shape_w_channel[0] if len(median_shape_w_channel) == 4 else 1

    dataset_config = {
        'median_shape': tuple(int(x) for x in median_shape),
        'min_shape': tuple(int(x) for x in min_shape),
        'max_shape': tuple(int(x) for x in max_shape),
        'median_spacing': [float(x) for x in median_spacing],
        'image_channel_mins': [float(x) for x in img_global_channel_min],
        'image_channel_maxs': [float(x) for x in img_global_channel_max],
        'dose_channel_mins': [float(x) for x in dose_global_channel_min],
        'dose_channel_maxs': [float(x) for x in dose_global_channel_max],
        'n_classes': int(len(unique_labels)),
        'class_labels': [int(c) for c in unique_labels],
        'n_channels': int(n_channels),
        'n_patients': int(n_patients)
    }
    with open(os.path.join(dataset_save_path, 'dataset.json'), 'w') as f:
        json.dump(dataset_config, f, indent=4)

    print(f"\nDataset configuration file saved in {os.path.join(dataset_save_path, 'dataset.json')}")

    print(f"\nConfiguring image generation parameters for Dataset ID: {formatted_task_number}")

    image_channels = [i for i in range(dataset_config['n_channels'])]

    if 0.7 * dataset_config['n_patients'] < 100:
        n_epochs_multiplier = 1
    elif 100 < 0.7 * dataset_config['n_patients'] < 500:
        n_epochs_multiplier = 2
    else:
        n_epochs_multiplier = 3

    vae_dict_2d = create_autoencoder_dict(dataset_config, image_channels, spatial_dims=2)
    vae_dict_3d = create_autoencoder_dict(dataset_config, image_channels, spatial_dims=3)

    ddpm_dict_2d = create_ddpm_dict(dataset_config, spatial_dims=2)
    ddpm_dict_3d = create_ddpm_dict(dataset_config, spatial_dims=3)

    config_2d = create_config_dict(dataset_config, image_channels, n_epochs_multiplier, vae_dict_2d, ddpm_dict_2d)
    config_3d = create_config_dict(dataset_config, image_channels, n_epochs_multiplier, vae_dict_3d, ddpm_dict_3d)

    print('\nConfiguring batch size and gradient accumulation steps based on GPU capacity...')
    batch_size_2d, grad_accumulate_step_2d = auto_select_hyperparams(formatted_task_number, AutoEncoder, config_2d, model_type='2d', init_batch_size=24, init_grad_accum=1)
    batch_size_3d, grad_accumulate_step_3d = auto_select_hyperparams(formatted_task_number, AutoEncoder, config_3d, model_type='3d', init_batch_size=2, init_grad_accum=1)

    config_2d['ae_batch_size'] = batch_size_2d
    config_2d['ddpm_batch_size'] = batch_size_2d
    config_2d['grad_accumulate_step'] = grad_accumulate_step_2d

    config_3d['ae_batch_size'] = batch_size_3d
    config_3d['ddpm_batch_size'] = batch_size_3d * 2
    config_3d['grad_accumulate_step'] = grad_accumulate_step_3d

    config = {'2d': config_2d, '3d': config_3d}

    config_save_path = os.path.join(dataset_save_path, 'medimgen_config.yaml')

    # Custom Dumper to avoid anchors and enforce list formatting
    class CustomDumper(yaml.SafeDumper):
        def ignore_aliases(self, data):
            return True  # Removes YAML anchors (&id001)

    # Ensure lists stay in flow style
    def represent_list(dumper, data):
        return dumper.represent_sequence('tag:yaml.org,2002:seq', data, flow_style=True)

    CustomDumper.add_representer(list, represent_list)

    # Save to YAML with all fixes
    with open(config_save_path, "w") as file:
        yaml.dump(config, file, sort_keys=False, Dumper=CustomDumper)

    print(f"Experiment configuration file saved at {config_save_path}")




