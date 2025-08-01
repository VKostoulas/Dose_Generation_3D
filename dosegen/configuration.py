import json
import os
import glob
import zarr
import yaml
import argparse
import nibabel as nib
import numpy as np

from zarr.codecs import BloscCodec, BloscShuffle
from concurrent.futures import ProcessPoolExecutor


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


def get_ae_n_layers(max_image_size):
    # use maximum of 3 autoencoder downsampling layers
    # we want the latent dims to be less than 100 to be manageable (say less than 96)
    if np.max(max_image_size) <= 96:
        ae_n_layers = 1
    elif np.max(max_image_size) <= 384:
        ae_n_layers = 2
    else:
        ae_n_layers = 3

    return ae_n_layers


def get_patch_size(max_image_size):
    # For each max dim, find the closest integer smaller than dim which can be divided by 2**(ae_n_layers + 2)
    # (number of autoencoder downsample layers and 2 downsample layers from the DDPM)
    ae_n_layers = get_ae_n_layers(max_image_size)

    patch_size = []
    for size in max_image_size:
        while size % 2**(ae_n_layers + 2) != 0:
            size -= 1
        patch_size.append(size)

    return patch_size


def create_vae_dict(max_image_size, spatial_dims):
    vae_n_layers = get_ae_n_layers(max_image_size)
    base_autoencoder_channels = [64, 128, 256, 256] if spatial_dims == 2 else [32, 64, 128, 128]

    vae_dict = {'spatial_dims': spatial_dims,
                'latent_channels': 8,
                'num_res_blocks': 2,
                'with_encoder_nonlocal_attn': False,
                'with_decoder_nonlocal_attn': False,
                'use_flash_attention': False,
                'use_checkpointing': False,
                'use_convtranspose': False,
                'num_channels': base_autoencoder_channels[:vae_n_layers + 1],
                'attention_levels': [False] * (vae_n_layers + 1),
                'norm_num_groups': 16}

    return vae_dict


def create_vqvae_dict(max_image_size, spatial_dims):
    vqvae_n_layers = get_ae_n_layers(max_image_size)
    base_autoencoder_channels = [64, 128, 256, 256] if spatial_dims == 2 else [32, 64, 128, 128]

    vqvae_dict = {'spatial_dims': spatial_dims,
                  'channels': base_autoencoder_channels[:vqvae_n_layers + 1],
                  'num_res_layers': 2,
                  'num_res_channels': base_autoencoder_channels[:vqvae_n_layers + 1],
                  'num_embeddings': 8,
                  'embedding_dim': 128,
                  'commitment_cost': 0.25,
                  'use_checkpointing': False}
    return vqvae_dict


def create_ddpm_dict(spatial_dims):
    ddpm_dict = {'spatial_dims': spatial_dims,
                 'in_channels': 8,
                 'out_channels': 8,
                 'num_res_blocks': 2,
                 'use_flash_attention': False,
                 'channels': [256, 512, 768],
                 'attention_levels': [False, True, True],
                 'num_head_channels': [0, 512, 768]}
    return ddpm_dict


def create_config_dict(dataset_config, vae_dict, vqvae_dict, ddpm_dict, spatial_dims):
    patch_size = get_patch_size(dataset_config['max_shape'])
    patch_size = patch_size[1:] if spatial_dims == 2 else patch_size

    print(f"Patch size {spatial_dims}D: {patch_size}")

    ae_transformations = {
        "patch_size": patch_size,
        "scaling": True,
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
    ddpm_transformations = {
        "patch_size": patch_size,
        "scaling": True,
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

    if spatial_dims == 2:
        perceptual_params = {'spatial_dims': 2, 'network_type': "vgg"}
    else:
        perceptual_params = {'spatial_dims': 3, 'network_type': "vgg", 'is_fake_3d': True, 'fake_3d_ratio': 0.2}

    discriminator_params = {'spatial_dims': spatial_dims, 'out_channels': 1, 'num_channels': 64, 'num_layers_d': 3}

    # adjust parameters based on the training model (2d/3d) and number of training data
    n_epochs = 300 if spatial_dims == 3 else 200
    n_epochs = n_epochs * 2 if dataset_config['n_patients'] > 300 else n_epochs

    ae_batch_size = 24 if spatial_dims == 2 else 2
    ddpm_batch_size = 24 if ddpm_dict['spatial_dims'] == 2 else 4
    grad_accumulate_step = 1

    config = {
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
        'ae_learning_rate': 1e-4,
        'd_learning_rate': 1e-4,
        'autoencoder_warm_up_epochs': 5,
        'vae_params': vae_dict,
        'vqvae_params': vqvae_dict,
        'perceptual_params': perceptual_params,
        'discriminator_params': discriminator_params,
        'ddpm_learning_rate': 5e-5,
        'ddpm_params': ddpm_dict,
        'adv_weight': 0.01,
        'perc_weight': 0.5 if spatial_dims == 2 else 0.125,
        'kl_weight': 1e-6 if spatial_dims == 2 else 1e-7,
        'q_weight': 1
    }
    return config


def get_min_max_per_channel(image):
    min_max_per_channel = []
    for c in range(image.shape[0]):
        channel_data = image[c]
        vmin = np.min(channel_data)
        vmax = np.max(channel_data)
        min_max_per_channel.append((vmin, vmax))
    return min_max_per_channel


def normalize_dose_w_global_minmax(dose, dose_global_minmax):
    return (dose - dose_global_minmax[0]) / (dose_global_minmax[1] - dose_global_minmax[0])


def normalize_zscore_minmax(image):
    normalized = np.zeros_like(image, dtype=np.float32)
    stats_per_channel = []
    for c in range(image.shape[0]):
        channel_data = image[c]
        c_mean = np.mean(channel_data)
        c_std = np.std(channel_data)
        z_image = (channel_data - c_mean) / c_std
        z_min = np.min(z_image)
        z_max = np.max(z_image)
        normalized[c] = (z_image - z_min) / (z_max - z_min)
        stats_per_channel.append((c_mean, c_std, z_min, z_max))
    return normalized, stats_per_channel


def clean_numpy_scalars(obj):
    if isinstance(obj, np.generic):
        return obj.item()  # Convert np.float64, np.int64, etc. to Python scalars
    elif isinstance(obj, tuple):
        return tuple(clean_numpy_scalars(x) for x in obj)
    elif isinstance(obj, list):
        return [clean_numpy_scalars(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: clean_numpy_scalars(v) for k, v in obj.items()}
    else:
        return obj


def process_patient(patient_id, patient_dict, save_path, median_shape, dose_global_minmax):
    log_lines = [f"Processing {patient_id}..."]

    file_save_path = os.path.join(save_path, patient_id + '.zarr')

    image = nib.load(patient_dict[patient_id]['image'])
    dose = nib.load(patient_dict[patient_id]['dose'])
    label = nib.load(patient_dict[patient_id]['label'])

    image = image.get_fdata()
    dose = dose.get_fdata()
    label = label.get_fdata()

    if image.ndim == 3:
        image = np.expand_dims(image, axis=-1)
    image = np.transpose(image, (3, 2, 1, 0))

    image_min_max = clean_numpy_scalars(get_min_max_per_channel(image))
    image, norm_image_stats = normalize_zscore_minmax(image)
    log_lines.append("    Image Min-Max before and after normalization:")
    log_lines.append(f"        before: {image_min_max} -- after: {clean_numpy_scalars(get_min_max_per_channel(image))}")

    dose = np.transpose(dose, (2, 1, 0))
    dose_min_max = clean_numpy_scalars((np.min(dose), np.max(dose)))
    dose = normalize_dose_w_global_minmax(dose, dose_global_minmax)
    log_lines.append("    Dose Min-Max before and after normalization:")
    log_lines.append(f"        before: {dose_min_max} -- after: {clean_numpy_scalars((np.min(dose), np.max(dose)))}")

    label = np.transpose(label, (2, 1, 0))
    unique_labels = np.unique(label).tolist()
    class_locations = get_sampled_class_locations(label, samples_per_slice=50)

    properties = {'class_locations': class_locations, 'image_min_max': image_min_max, 'norm_image_stats': norm_image_stats}

    compressor = BloscCodec(cname='zstd', clevel=3, shuffle=BloscShuffle.bitshuffle)
    image_chunks = (median_shape[0], 1) + tuple(median_shape[-2:])
    label_or_dose_chunks = (1,) + tuple(median_shape[-2:])
    z_file = zarr.open(file_save_path, mode='w')
    z_file.create_array(name='image', data=image.astype(np.float32), chunks=image_chunks, compressors=compressor, overwrite=True)
    z_file.create_array(name='dose', data=dose.astype(np.float32), chunks=label_or_dose_chunks, compressors=compressor, overwrite=True)
    z_file.create_array(name='label', data=label.astype(np.uint8), chunks=label_or_dose_chunks, compressors=compressor, overwrite=True)
    z_file.attrs['properties'] = clean_numpy_scalars(properties)

    log_lines.append(f"    Saved processed image, dose, label and properties to {file_save_path}")

    return {
        "patient_id": patient_id,
        "shape": image.shape,
        "labels": [item for item in unique_labels if item != 0],
        "log": "\n".join(log_lines)
    }


def process_patient_wrapper(args):
    return process_patient(*args)


def get_image_shape_and_image_dose_min_max(patient_paths):
    image = nib.load(patient_paths['image'])
    dose = nib.load(patient_paths['dose'])

    image = image.get_fdata()
    dose = dose.get_fdata()

    if image.ndim == 3:
        image = np.expand_dims(image, axis=-1)
    image = np.transpose(image, (3, 2, 1, 0))

    img_min_max_per_channel = get_min_max_per_channel(image)

    if dose.ndim == 3:
        dose = np.expand_dims(dose, axis=-1)
    dose = np.transpose(dose, (3, 2, 1, 0))

    dose_min_max = get_min_max_per_channel(dose)
    assert len(dose_min_max) == 1
    dose_min_max = dose_min_max[0]

    return image.shape, img_min_max_per_channel, dose_min_max


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


def extract_spacing(patient_paths):
    path = patient_paths['image']
    img = nib.load(path)
    spacing = np.sqrt(np.sum(img.affine[:3, :3] ** 2, axis=0))  # voxel spacing from affine
    return spacing


def calculate_median_spacing(patient_dict):
    with ProcessPoolExecutor() as executor:
        image_spacings = list(executor.map(extract_spacing, patient_dict.values()))

    return tuple(np.median(image_spacings, axis=0).tolist())


def calculate_dataset_shapes_and_channel_min_max(patient_dict):

    with ProcessPoolExecutor() as executor:
        results = list(executor.map(get_image_shape_and_image_dose_min_max, patient_dict.values()))

    shapes, img_min_max_per_channel, dose_min_max = zip(*results)
    median_shape = tuple(np.median(np.array(shapes), axis=0).astype(int).tolist())
    min_shape = tuple(np.min(np.array(shapes), axis=0).astype(int).tolist())
    max_shape = tuple(np.max(np.array(shapes), axis=0).astype(int).tolist())

    img_min_max_per_channel = np.array(img_min_max_per_channel)  # Shape: (num_images, num_channels, 2)
    img_global_channel_min = tuple(img_min_max_per_channel[..., 0].min(axis=0).tolist())
    img_global_channel_max = tuple(img_min_max_per_channel[..., 1].max(axis=0).tolist())

    dose_min_max = np.array(dose_min_max)  # Shape: (num_images, 2)
    dose_global_min = dose_min_max[:, 0].min()
    dose_global_max = dose_min_max[:, 1].max()
    dose_global_minmax = tuple(float(x) for x in (dose_global_min, dose_global_max))

    return median_shape, min_shape, max_shape, img_global_channel_min, img_global_channel_max, dose_global_minmax


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
    data_save_path = os.path.join(dataset_save_path, 'data')

    if os.path.exists(dataset_save_path):
        raise FileExistsError(f"Dataset {basename} already exists.")

    # Given dataset folder must have a name in the form TaskXXX_DatasetName
    # The dataset folder should always contain an 'images' folder, a 'doses' folder and a 'labels' folder
    # All the files should be .nii.gz files

    os.makedirs(data_save_path, exist_ok=True)

    image_paths = glob.glob(os.path.join(dataset_path, 'images') + "/*.nii.gz")
    patient_ids = sorted([os.path.basename(path).replace('.nii.gz', '') for path in image_paths])
    patient_dict = {}
    for pid in patient_ids:
        img_path = os.path.join(dataset_path, 'images', f"{pid}.nii.gz")
        dose_path = os.path.join(dataset_path, 'doses', f"{pid}.nii.gz")
        label_path = os.path.join(dataset_path, 'labels', f"{pid}.nii.gz")
        patient_dict[pid] = {'image': img_path, 'dose': dose_path, 'label': label_path}

    print(f"\nNumber of patients: {len(patient_ids)}")
    print("\nCalculating median voxel spacing of the whole dataset...")
    median_spacing = calculate_median_spacing(patient_dict)
    print("Calculating dataset min/max values, median, min and max shape...")
    dataset_results = calculate_dataset_shapes_and_channel_min_max(patient_dict)
    median_shape, min_shape, max_shape, img_global_channel_min, img_global_channel_max, dose_global_minmax = dataset_results
    print(f"\nMedian voxel spacing: {median_spacing}")
    print(f"Median Shape: {median_shape}")
    print(f"Min Shape: {min_shape}")
    print(f"Max Shape: {max_shape}")
    print(f"Image Min per channel: {img_global_channel_min}")
    print(f"Image Max per channel: {img_global_channel_max}")
    print(f"Dose Min/Max: {dose_global_minmax}\n")

    median_shape_w_channel = median_shape
    median_shape, min_shape, max_shape = median_shape[1:], min_shape[1:], max_shape[1:]

    results = []
    args_list = [(pid, patient_dict, data_save_path, median_shape, dose_global_minmax)
                 for pid in patient_ids]

    with ProcessPoolExecutor() as executor:
        for result in executor.map(process_patient_wrapper, args_list):
            print(result["log"])
            results.append(result)

    all_labels = [lbl for r in results for lbl in r["labels"]]
    unique_labels = sorted(set(all_labels))
    n_patients = len(results)
    n_img_channels = median_shape_w_channel[0] if len(median_shape_w_channel) == 4 else 1

    dataset_config = {
        'median_shape': tuple(int(x) for x in median_shape),
        'min_shape': tuple(int(x) for x in min_shape),
        'max_shape': tuple(int(x) for x in max_shape),
        'median_spacing': tuple(float(x) for x in median_spacing),
        'image_channel_mins': tuple(float(x) for x in img_global_channel_min),
        'image_channel_maxs': tuple(float(x) for x in img_global_channel_max),
        'dose_min_max': dose_global_minmax,
        'n_classes': int(len(unique_labels)),
        'class_labels': tuple(int(c) for c in unique_labels),
        'n_img_channels': int(n_img_channels),
        'n_patients': int(n_patients)
    }
    with open(os.path.join(dataset_save_path, 'dataset.json'), 'w') as f:
        json.dump(dataset_config, f, indent=4)

    print(f"\nDataset configuration file saved in {os.path.join(dataset_save_path, 'dataset.json')}")

    print(f"\nConfiguring image generation parameters for Dataset ID: {formatted_task_number}")

    vae_dict_2d = create_vae_dict(dataset_config['max_shape'], spatial_dims=2)
    vae_dict_3d = create_vae_dict(dataset_config['max_shape'], spatial_dims=3)

    vqvae_dict_2d = create_vqvae_dict(dataset_config['max_shape'], spatial_dims=2)
    vqvae_dict_3d = create_vqvae_dict(dataset_config['max_shape'], spatial_dims=3)

    ddpm_dict_2d = create_ddpm_dict(spatial_dims=2)
    ddpm_dict_3d = create_ddpm_dict(spatial_dims=3)

    config_2d = create_config_dict(dataset_config, vae_dict_2d, vqvae_dict_2d, ddpm_dict_2d, spatial_dims=2)
    config_3d = create_config_dict(dataset_config, vae_dict_3d, vqvae_dict_3d, ddpm_dict_3d, spatial_dims=3)

    config = {'2d': config_2d, '3d': config_3d}

    config_save_path = os.path.join(dataset_save_path, 'dosegen_config.yaml')

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




