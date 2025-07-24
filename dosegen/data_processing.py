import os
import torch
import glob
import json
import zarr
import blosc2
import pickle
import multiprocessing
import numpy as np
import torch.nn.functional as F

from typing import Tuple
from typing import Union, List
from functools import partial
from batchgenerators.utilities.file_and_folder_operations import subfiles
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import KFold, train_test_split
# from acvl_utils.cropping_and_padding.bounding_boxes import crop_and_pad_nd
from batchgenerators.augmentations.utils import rotate_coords_3d, rotate_coords_2d
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform


def generate_crossval_split(train_identifiers, seed=12345, n_splits=5):
    splits = []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (train_idx, test_idx) in enumerate(kfold.split(train_identifiers)):
        train_keys = np.array(train_identifiers)[train_idx]
        test_keys = np.array(train_identifiers)[test_idx]
        splits.append({})
        splits[-1]['train'] = list(train_keys)
        splits[-1]['val'] = list(test_keys)
    return splits


def create_split_files(dataset_id, splitting, model_type, seed=12345):
    """
    Creates and saves split files for a given dataset.

    Parameters:
    - dataset_id (str): Dataset identifier (e.g., '001', '002').
    - splitting (str): Type of split ('train-val-test' or '5-fold').
    - model_type (str): Model type ('2d' or '3d').
    - seed (int, optional): Random seed for reproducibility. Default is 12345.
    """

    # preprocessed_dataset_path = glob.glob(os.getenv('nnUNet_preprocessed') + f'/Dataset{dataset_id}*/')[0]
    preprocessed_dataset_path = glob.glob(os.getenv('medimgen_preprocessed') + f'/Task{dataset_id}*/')[0]
    # nnunet_path = "nnUNetPlans_2d" if model_type == '2d' else "nnUNetPlans_3d_fullres"
    # dataset_path = os.path.join(preprocessed_dataset_path, nnunet_path)
    dataset_path = os.path.join(preprocessed_dataset_path, 'imagesTr')

    split_file_name = "splits_train_val_test.json" if splitting == "train-val-test" else "splits_final.json"
    split_file_path = os.path.join(preprocessed_dataset_path, split_file_name)

    if os.path.exists(split_file_path):
        print(f"Split file already exists at {split_file_path}. Using this for training.")
        return split_file_path

    file_paths = glob.glob(os.path.join(dataset_path, "*.zarr"))
    file_names = [os.path.basename(fp).replace('.zarr', '') for fp in file_paths]
    if not file_names:
        # maybe we got .npz files
        file_paths = glob.glob(os.path.join(dataset_path, "*.npz"))
        file_names = [os.path.basename(fp).replace('.npz', '') for fp in file_paths]
        if not file_names:
            # we got .b2nd files
            file_paths = glob.glob(os.path.join(dataset_path, "*.b2nd"))
            file_names = [os.path.basename(fp).replace('.b2nd', '') for fp in file_paths if '_seg' not in fp]

    if splitting == "train-val-test":
        # Split data into 70% training, 10% validation, and 20% testing
        train_val, test = train_test_split(file_names, test_size=0.2, random_state=seed)
        train, val = train_test_split(train_val, test_size=0.125, random_state=seed)  # 10% of total data
        split_data = {"train": train, "val": val, "test": test}
    elif splitting == "5-fold":
        split_data = generate_crossval_split(file_names, seed=seed, n_splits=5)
    else:
        raise ValueError("Invalid splitting option. Choose 'train-val-test' or '5-fold'.")

    # Save the split dictionary as a pickle file
    with open(split_file_path, 'w') as f:
        json.dump(split_data, f, indent=4)

    print(f"{splitting} splitting file saved at {split_file_path}")
    return split_file_path


def get_data_ids(split_file_path, fold=None):

    with open(split_file_path, 'r') as f:
        split_data = json.load(f)

    if fold is not None:
        train_ids = split_data[int(fold)]['train']
        val_ids = split_data[int(fold)]['val']
    else:
        train_ids = split_data['train']
        val_ids = split_data['val']

    print(f"{len(train_ids)} patients for training")
    print(f"{len(val_ids)} patients for validation")
    return {"train": train_ids, "val": val_ids}


def get_data_loaders(config, dataset_id, splitting, batch_size, model_type, transformations, fold=None):
    # based on input arg splitting, the dataloader will return 2 different pairs of train-val loaders:
    # splitting: "train-val-test"
    #        train loader will contain 70% and val loader 10% of the whole dataset. 20% left fot test set
    # splitting: "5-fold"
    #        argument fold must be specified:
    #        based on fold argument the train-val loaders will contain the 80-20% ratio specified in the
    #        5-fold splitting for this fold

    split_file_path = create_split_files(dataset_id, splitting, model_type, seed=12345)
    data_ids = get_data_ids(split_file_path, fold)

    # preprocessed_dataset_path = glob.glob(os.getenv('nnUNet_preprocessed') + f'/Dataset{dataset_id}*/')[0]
    preprocessed_dataset_path = glob.glob(os.getenv('medimgen_preprocessed') + f'/Task{dataset_id}*/')[0]
    # nnunet_path = "nnUNetPlans_2d" if model_type == '2d' else "nnUNetPlans_3d_fullres"
    # dataset_path = os.path.join(preprocessed_dataset_path, nnunet_path)
    dataset_path = os.path.join(preprocessed_dataset_path, 'imagesTr')

    train_ds = MedicalDataset(data_path=dataset_path, data_ids=data_ids['train'], batch_size=batch_size,
                              section="training", transformation_args=transformations,
                              oversample_foreground_percent=config['oversample_ratio'], channel_ids=config['input_channels'])
    val_ds = MedicalDataset(data_path=dataset_path, data_ids=data_ids['val'], batch_size=batch_size,
                            section="validation", transformation_args=transformations,
                            oversample_foreground_percent=config['oversample_ratio'], channel_ids=config['input_channels'])

    train_sampler = CustomBatchSampler(train_ds, batch_size=batch_size, number_of_steps=250, shuffle=True)
    val_sampler = CustomBatchSampler(val_ds, batch_size=batch_size, number_of_steps=50, shuffle=False)
    loader_args = dict(num_workers=config['num_workers'], pin_memory=True, prefetch_factor=2)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **loader_args)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler,**loader_args)
    return train_loader, val_loader


def crop_and_pad_nd(
        image: Union[torch.Tensor, np.ndarray, blosc2.ndarray.NDArray],
        bbox: List[List[int]],
        pad_value = 0
) -> Union[torch.Tensor, np.ndarray]:
    """
    Crops a bounding box directly specified by bbox, excluding the upper bound.
    If the bounding box extends beyond the image boundaries, the cropped area is padded
    to maintain the desired size. Initial dimensions not included in bbox remain unaffected.

    Parameters:
    - image: N-dimensional torch.Tensor or np.ndarray representing the image
    - bbox: List of [[dim_min, dim_max], ...] defining the bounding box for the last dimensions.

    Returns:
    - Cropped and padded patch of the requested bounding box size, as the same type as `image`.
    """

    # Determine the number of dimensions to crop based on bbox
    crop_dims = len(bbox)
    img_shape = image.shape
    num_dims = len(img_shape)

    # Initialize the crop and pad specifications for each dimension
    slices = []
    padding = []
    output_shape = list(img_shape[:num_dims - crop_dims])  # Initial dimensions remain as in the original image
    target_shape = output_shape + [max_val - min_val for min_val, max_val in bbox]

    # Iterate through dimensions, applying bbox to the last `crop_dims` dimensions
    for i in range(num_dims):
        if i < num_dims - crop_dims:
            # For initial dimensions not covered by bbox, include the entire dimension
            slices.append(slice(None))
            padding.append([0, 0])
            output_shape.append(img_shape[i])  # Keep the initial dimensions as they are
        else:
            # For dimensions specified in bbox, directly use the min and max bounds
            dim_idx = i - (num_dims - crop_dims)  # Index within bbox

            min_val = bbox[dim_idx][0]
            max_val = bbox[dim_idx][1]

            # Check if the bounding box is completely outside the image bounds
            if max_val <= 0 or min_val >= img_shape[i]:
                # If outside bounds, return an empty array or tensor of the target shape
                if isinstance(image, torch.Tensor):
                    return torch.zeros(target_shape, dtype=image.dtype, device=image.device)
                elif isinstance(image, (np.ndarray, blosc2.ndarray.NDArray, zarr.core.Array)):
                    return np.zeros(target_shape, dtype=image.dtype)

            # Calculate valid cropping ranges within image bounds, excluding the upper bound
            valid_min = max(min_val, 0)
            valid_max = min(max_val, img_shape[i])  # Exclude upper bound by using max_val directly
            slices.append(slice(valid_min, valid_max))

            # Calculate padding needed for this dimension
            pad_before = max(0, -min_val)
            pad_after = max(0, max_val - img_shape[i])
            padding.append([pad_before, pad_after])

            # Define the shape based on the bbox range in this dimension
            output_shape.append(max_val - min_val)

    # Crop the valid part of the bounding box
    cropped = image[tuple(slices)]

    # Apply padding to the cropped patch
    if isinstance(image, torch.Tensor):
        flattened_padding = [p for sublist in reversed(padding) for p in sublist]  # Flatten in reverse order for PyTorch
        padded_cropped = F.pad(cropped, flattened_padding, mode="constant", value=pad_value)
    elif isinstance(image, (np.ndarray, blosc2.ndarray.NDArray, zarr.core.Array)):
        pad_width = [(p[0], p[1]) for p in padding]
        padded_cropped = np.pad(cropped, pad_width=pad_width, mode='constant', constant_values=pad_value)
    else:
        raise ValueError(f'Unsupported image type {type(image)}')

    return padded_cropped


# TODO: Fix this?
def _convert_to_npy(npz_file: str, unpack_segmentation: bool = True, overwrite_existing: bool = False,
                    verify_npy: bool = False, fail_ctr: int = 0) -> None:
    data_npy = npz_file[:-3] + "npy"
    seg_npy = npz_file[:-4] + "_seg.npy"
    try:
        npz_content = None  # will only be opened on demand

        if overwrite_existing or not os.path.isfile(data_npy):
            try:
                npz_content = np.load(npz_file) if npz_content is None else npz_content
            except Exception as e:
                print(f"Unable to open preprocessed file {npz_file}. Rerun nnUNetv2_preprocess!")
                raise e
            np.save(data_npy, np.ascontiguousarray(npz_content['data']))

        if unpack_segmentation and (overwrite_existing or not os.path.isfile(seg_npy)):
            try:
                npz_content = np.load(npz_file) if npz_content is None else npz_content
            except Exception as e:
                print(f"Unable to open preprocessed file {npz_file}. Rerun nnUNetv2_preprocess!")
                raise e
            np.save(npz_file[:-4] + "_seg.npy", np.ascontiguousarray(npz_content['seg']))

        if verify_npy:
            try:
                np.load(data_npy, mmap_mode='r')
                if os.path.isfile(seg_npy):
                    np.load(seg_npy, mmap_mode='r')
            except ValueError:
                os.remove(data_npy)
                os.remove(seg_npy)
                print(f"Error when checking {data_npy} and {seg_npy}, fixing...")
                if fail_ctr < 2:
                    _convert_to_npy(npz_file, unpack_segmentation, overwrite_existing, verify_npy, fail_ctr+1)
                else:
                    raise RuntimeError("Unable to fix unpacking. Please check your system or rerun nnUNetv2_preprocess")

    except KeyboardInterrupt:
        if os.path.isfile(data_npy):
            os.remove(data_npy)
        if os.path.isfile(seg_npy):
            os.remove(seg_npy)
        raise KeyboardInterrupt


class MedicalDataset(Dataset):
    def __init__(self, data_path, data_ids, batch_size, section, transformation_args, oversample_foreground_percent,
                 channel_ids=None, probabilistic_oversampling=False):
        self.data_path = data_path
        self.ids = data_ids
        self.batch_size = batch_size
        self.section = section
        self.transformation_args = transformation_args
        self.oversample_foreground_percent = oversample_foreground_percent
        self.channel_ids = channel_ids

        self.patch_size = transformation_args["patch_size"]

        augmentation_params = self.configure_augmentation_params(heavy_augmentation=False)
        self.initial_patch_size = augmentation_params['initial_patch_size'] if section == 'training' else self.patch_size
        self.transformation_args['rot_for_da'] = augmentation_params['rot_for_da'] if transformation_args['rotation'] else None
        self.transformation_args['dummy_2d'] = augmentation_params['do_dummy_2d'] if transformation_args['dummy_2d'] else None
        self.transformation_args['mirror_axes'] = augmentation_params['mirror_axes'] if transformation_args['mirror'] else None
        self.transformation_args['scaling_range'] = augmentation_params['scale_range'] if transformation_args['scaling'] else None
        self.transformation_args['brightness_range'] = augmentation_params['brightness_range'] if transformation_args['brightness'] else None
        self.transformation_args['contrast_range'] = augmentation_params['contrast_range'] if transformation_args['contrast'] else None
        self.transformation_args['gamma_range'] = augmentation_params['gamma_range'] if transformation_args['gamma'] else None

        # If we get a 2D patch size, make it pseudo 3D and remember to remove the singleton dimension before
        # returning the batch
        self.patch_size = (1, *self.patch_size) if len(self.patch_size) == 2 else self.patch_size
        self.initial_patch_size = (1, *self.initial_patch_size) if len(self.initial_patch_size) == 2 else self.initial_patch_size

        self.need_to_pad = (np.array(self.initial_patch_size) - np.array(self.patch_size)).astype(int)
        self.oversampling_method = self._oversample_last_XX_percent if not probabilistic_oversampling \
            else self._probabilistic_oversampling

        validation = False if self.section == "training" else True
        self.transformations = define_nnunet_transformations(self.transformation_args, validation)

    def __len__(self):
        return len(self.ids)

    def unpack_dataset(self, unpack_segmentation=False, overwrite_existing=False, num_processes=8, verify=False):
        npz_files = glob.glob(self.data_path + '*.npz')
        if len(npz_files) > 0:
            print("Unpacking dataset...")
            with multiprocessing.get_context("spawn").Pool(num_processes) as p:
                npz_files = subfiles(self.data_path, True, None, ".npz", True)
                p.starmap(_convert_to_npy, zip(npz_files,
                                               [unpack_segmentation] * len(npz_files),
                                               [overwrite_existing] * len(npz_files),
                                               [verify] * len(npz_files))
                          )

    def pack_dataset(self):
        npy_files = glob.glob(self.data_path + '*.npy')
        if len(npy_files) > 0:
            npy_files_removed = 0
            for filename in os.listdir(self.data_path):
                if filename.endswith('.npy'):
                    file_path = os.path.join(self.data_path, filename)
                    try:
                        os.remove(file_path)
                        npy_files_removed += 1
                    except Exception as e:
                        print(f"Error removing {file_path}: {e}")
            print(f"{npy_files_removed} .npy files were deleted")

    # from nnunet
    def get_initial_patch_size(self, rot_x, rot_y, rot_z, scale_range):
        dim = len(self.patch_size)

        # Ensure rotation angles are always within reasonable bounds (max 90 degrees)
        rot_x = min(np.pi / 2, max(np.abs(rot_x)) if isinstance(rot_x, (tuple, list)) else rot_x)
        rot_y = min(np.pi / 2, max(np.abs(rot_y)) if isinstance(rot_y, (tuple, list)) else rot_y)
        rot_z = min(np.pi / 2, max(np.abs(rot_z)) if isinstance(rot_z, (tuple, list)) else rot_z)

        coords = np.array(self.patch_size[-dim:])
        final_shape = np.copy(coords)
        # Apply rotations along each axis and update final shape
        if len(coords) == 3:
            final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, rot_x, 0, 0)), final_shape)), 0)
            final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, rot_y, 0)), final_shape)), 0)
            final_shape = np.max(np.vstack((np.abs(rotate_coords_3d(coords, 0, 0, rot_z)), final_shape)), 0)
        elif len(coords) == 2:
            final_shape = np.max(np.vstack((np.abs(rotate_coords_2d(coords, rot_x)), final_shape)), 0)

        # Adjust the patch size based on the minimum scaling factor
        final_shape /= min(scale_range)
        return final_shape.astype(int)

    # from nnunet adapted
    def configure_augmentation_params(self, heavy_augmentation=False):
        """
        Configures rotation-based data augmentation, determines if 2D augmentation is needed,
        and computes the initial patch size to accommodate transformations.
        """
        anisotropy_threshold = 3
        dim = len(self.patch_size)

        # do what nnU-Net does
        if heavy_augmentation:

            if dim == 2:
                do_dummy_2d_data_aug = False
                rotation_for_DA = (-np.pi * 15 / 180, np.pi * 15 / 180) if max(self.patch_size) / min(self.patch_size) > 1.5 else (-np.pi, np.pi)
                mirror_axes = (0, 1)
            elif dim == 3:
                # Determine if 2D augmentation should be used (for highly anisotropic data)
                do_dummy_2d_data_aug = (max(self.patch_size) / self.patch_size[0]) > anisotropy_threshold
                # Set rotation ranges based on augmentation type
                rotation_for_DA = (-np.pi, np.pi) if do_dummy_2d_data_aug else (-np.pi * 30 / 180, np.pi * 30 / 180)
                mirror_axes = (0, 1, 2)
            else:
                raise ValueError("Invalid patch size dimensionality. Must be 2D or 3D.")

            # Compute the initial patch size, adjusting for rotation and scaling
            initial_patch_size = self.get_initial_patch_size(rot_x=rotation_for_DA, rot_y=rotation_for_DA,
                                                             rot_z=rotation_for_DA, scale_range=(0.7, 1.4))  # Standard scale range used in nnU-Net

            # If using 2D augmentation, force the depth dimension to remain unchanged
            if do_dummy_2d_data_aug:
                initial_patch_size[0] = self.patch_size[0]

            scale_range = (0.7, 1.4)
            brightness_range = (0.75, 1.25)
            contrast_range = (0.75, 1.25)
            gamma_range = (0.7, 1.5)

        # soft augmentation for image generation training
        else:
            # rotation around z axis
            def rot(rot_dim, image, dim):
                if dim == rot_dim:
                    return np.random.uniform(-0.174533, 0.174533)
                else:
                    return 0

            rot_dim = 0 if dim == 3 else 2 if dim == 2 else None
            rotation_for_DA = partial(rot, rot_dim)
            do_dummy_2d_data_aug = False
            initial_patch_size = self.patch_size
            mirror_axes = (2,) if dim == 3 else (1,)
            scale_range = (0.9, 1.1)
            brightness_range = (0.9, 1.1)
            contrast_range = (0.9, 1.1)
            gamma_range = (0.9, 1.1)

        augmentation_dict = {'rot_for_da': rotation_for_DA, 'do_dummy_2d': do_dummy_2d_data_aug,
                             'initial_patch_size': tuple(initial_patch_size), 'mirror_axes': mirror_axes,
                             'scale_range': scale_range, 'brightness_range': brightness_range,
                             'contrast_range': contrast_range, 'gamma_range': gamma_range}

        return augmentation_dict

    # from nnunet
    def _oversample_last_XX_percent(self, sample_idx: int) -> bool:
        """ Determines if the current patch should contain foreground. """
        return sample_idx >= round(self.batch_size * (1 - self.oversample_foreground_percent))

    # from nnunet
    def _probabilistic_oversampling(self, sample_idx: int) -> bool:
        """ Uses a probability threshold to oversample foreground patches. """
        return np.random.uniform() < self.oversample_foreground_percent

    # from nnunet
    # def get_bbox(self, data_shape, force_fg, class_locations):
    #     """ Determines a bounding box for cropping a patch, ensuring balanced sampling. """
    #     dim = len(data_shape)
    #     need_to_pad = self.need_to_pad.copy()
    #
    #     for d in range(dim):
    #         # if case_all_data.shape + need_to_pad is still < patch size we need to pad more! We pad on both sides always
    #         if need_to_pad[d] + data_shape[d] < self.initial_patch_size[d]:
    #             need_to_pad[d] = self.initial_patch_size[d] - data_shape[d]
    #
    #     # we can now choose the bbox from -need_to_pad // 2 to shape - patch_size + need_to_pad // 2. Here we
    #     # define what the upper and lower bound can be to then sample form them with np.random.randint
    #     lbs = [-need_to_pad[i] // 2 for i in range(dim)]
    #     ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.initial_patch_size[i] for i in range(dim)]
    #
    #     # Select a random location unless foreground oversampling is required
    #     if not force_fg:
    #         bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
    #     else:
    #         assert class_locations is not None, 'if force_fg is set class_locations cannot be None'
    #         eligible_classes_or_regions = [i for i in class_locations.keys() if len(class_locations[i]) > 0]
    #         if len(eligible_classes_or_regions) == 0:
    #             # this only happens if some image does not contain foreground voxels at all
    #             # If the image does not contain any foreground classes, we fall back to random cropping
    #             bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]
    #         else:
    #             selected_class = eligible_classes_or_regions[np.random.choice(len(eligible_classes_or_regions))]
    #             voxels_of_that_class = class_locations[selected_class]
    #             selected_voxel = voxels_of_that_class[np.random.choice(len(voxels_of_that_class))]
    #             # selected voxel is center voxel. Subtract half the patch size to get lower bbox voxel.
    #             # Make sure it is within the bounds of lb and ub
    #             # i + 1 because we have first dimension 0!
    #             bbox_lbs = [max(lbs[i], selected_voxel[i + 1] - self.initial_patch_size[i] // 2) for i in range(dim)]
    #
    #     bbox_ubs = [bbox_lbs[i] + self.initial_patch_size[i] for i in range(dim)]
    #     return bbox_lbs, bbox_ubs

    def get_bbox(self, data_shape, force_fg, class_locations, is_2d=False):
        """
        Computes a bounding box (lower and upper) for patch cropping.
        Always center crops in H and W (y and x), random/fg sampling for slice/depth (z).
        """
        dim = len(data_shape)
        need_to_pad = self.need_to_pad.copy()

        for d in range(dim):
            if need_to_pad[d] + data_shape[d] < self.initial_patch_size[d]:
                need_to_pad[d] = self.initial_patch_size[d] - data_shape[d]

        lbs = [-need_to_pad[i] // 2 for i in range(dim)]
        ubs = [data_shape[i] + need_to_pad[i] // 2 + need_to_pad[i] % 2 - self.initial_patch_size[i] for i in
               range(dim)]

        # Default random bbox_lbs
        bbox_lbs = [np.random.randint(lbs[i], ubs[i] + 1) for i in range(dim)]

        if force_fg and class_locations is not None:
            eligible_classes = [cls for cls in class_locations if len(class_locations[cls]) > 0]

            if eligible_classes:
                selected_class = np.random.choice(eligible_classes)
                voxels = class_locations[selected_class]
                selected_voxel = voxels[np.random.choice(len(voxels))]  # (0, z, y, x)

                for i in range(dim):
                    if is_2d and i == 0:
                        bbox_lbs[0] = selected_voxel[0]  # slice index
                    elif not is_2d:
                        # 3D: all dims available; use voxel for z, keep y/x random for now
                        bbox_lbs[i] = max(lbs[i], min(selected_voxel[i] - self.initial_patch_size[i] // 2, ubs[i]))
            # Else: fallback to original random values (already set)

        # Overwrite H and W (last 2 dims) to be center-cropped
        # for i in range(dim - 2, dim):
        #     center = data_shape[i] // 2
        #     bbox_lbs[i] = center - self.initial_patch_size[i] // 2
        for i in range(dim - 2, dim):
            crop_size = self.initial_patch_size[i]
            image_size = data_shape[i]

            center = image_size // 2

            if image_size < crop_size:
                # Center the crop, allow negative lb if needed (handled later with padding)
                bbox_lbs[i] = center - crop_size // 2
            else:
                max_offset = min(10, center - crop_size // 2, image_size - center - (crop_size - crop_size // 2))
                offset = np.random.randint(-max_offset, max_offset + 1) if max_offset > 0 else 0
                adjusted_center = center + offset
                bbox_lbs[i] = adjusted_center - crop_size // 2

        bbox_ubs = [bbox_lbs[i] + self.initial_patch_size[i] for i in range(dim)]
        return bbox_lbs, bbox_ubs

    def transform(self, image):
        transformed = self.transformations(image=image)
        transformed_image = transformed["image"]
        return transformed_image

    def load_image(self, name):
        dparams = {'nthreads': 1}

        # Try .zarr first
        zarr_path = os.path.join(self.data_path, name + '.zarr')
        if os.path.isdir(zarr_path):
            zgroup = zarr.open_group(zarr_path, mode='r')
            image = zgroup['image']
        else:
            # Fallback to npy, npz, or b2nd
            image_path_npy = os.path.join(self.data_path, name + '.npy')
            if not os.path.isfile(image_path_npy):
                image_path_npz = os.path.join(self.data_path, name + '.npz')
                if not os.path.isfile(image_path_npz):
                    data_b2nd_file = os.path.join(self.data_path, name + '.b2nd')
                    image = blosc2.open(urlpath=data_b2nd_file, mode='r', dparams=dparams, mmap_mode='r')
                else:
                    image = np.load(os.path.join(self.data_path, name + '.npz'))['data']
            else:
                image = np.load(image_path_npy, mmap_mode='r')

        with open(os.path.join(self.data_path, name + '.pkl'), mode='rb') as f:
            properties = pickle.load(f)

        return image, properties

    def __getitem__(self, indexes):
        batch_idx, sample_idx = indexes
        name = self.ids[sample_idx]
        image, properties = self.load_image(name)

        # Decide if oversampling foreground is needed
        force_fg = self.oversampling_method(batch_idx)

        # Get bounding box for cropping
        shape = image.shape[1:]  # Remove channel dimension
        bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'], is_2d=self.patch_size[0] == 1)
        bbox = [[i, j] for i, j in zip(bbox_lbs, bbox_ubs)]

        image = crop_and_pad_nd(image, bbox, 0)

        # Select specific channels if channel_ids is provided
        if self.channel_ids is not None:
            image = image[self.channel_ids, ...]
            # min_max = [min_max[i] for i in self.channel_ids]

        if len(image.shape) < len(self.patch_size) + 1:
            image = np.expand_dims(image, axis=0)  # add channel dimension
        # image = np.expand_dims(image, axis=0)  # add batch dimension

        # scale to 0-1 per channel based on the statistics of the entire volume
        # mins = np.array([mm[0] for mm in min_max], dtype=np.float32).reshape((-1,) + (1,) * (image.ndim - 1))
        # maxs = np.array([mm[1] for mm in min_max], dtype=np.float32).reshape((-1,) + (1,) * (image.ndim - 1))
        # image = (image - mins) / (maxs - mins)

        image = np.squeeze(image, axis=1) if self.patch_size[0] == 1 else image
        image = torch.as_tensor(image).float()
        image = image.contiguous()
        image = self.transform(image)

        image = torch.clamp(image, min=0.0, max=1.0)

        # image = torch.squeeze(image, dim=0)
        return {'id': name, 'image': image}


class CustomBatchSampler(Sampler):
    def __init__(self, dataset, batch_size, number_of_steps=250, shuffle=True):
        super().__init__()
        self.batch_size = batch_size
        self.number_of_steps = number_of_steps
        self.shuffle = shuffle
        self.indices = list(range(len(dataset)))
        self.sample_order = []  # This will store the order in which we sample

    def define_indices(self):
        """
        Creates a sampling order ensuring each sample is used once before repetition.
        """
        if self.shuffle:
            np.random.shuffle(self.indices)

        # Generate the order in which samples will be taken
        self.sample_order = []
        total_needed = self.number_of_steps * self.batch_size
        available = self.indices.copy()

        while len(self.sample_order) < total_needed:
            if len(available) < self.batch_size:
                # If fewer than a batch size remains, shuffle and reset available
                available = self.indices.copy()
                if self.shuffle:
                    np.random.shuffle(available)

            # Take batch_size elements from available
            self.sample_order.extend(available[:self.batch_size])
            available = available[self.batch_size:]

    def __iter__(self):
        self.define_indices()

        for step in range(self.number_of_steps):
            batch_start = step * self.batch_size
            sample_indices = self.sample_order[batch_start: batch_start + self.batch_size]
            batch = [(i, sample_idx) for i, sample_idx in enumerate(sample_indices)]
            yield batch

    def __len__(self):
        return self.number_of_steps


def collate_fn(batch):
    """Collate function to stack tensors from a list of (image_patch, label_patch) pairs."""
    images, labels = zip(*batch)  # Unpack list of tuples
    images = torch.stack([torch.tensor(img, dtype=torch.float32) for img in images])
    labels = torch.stack([torch.tensor(lbl, dtype=torch.long) for lbl in labels])
    return images, labels


# def define_nnunet_transformations(params, validation=False, border_val_seg=-1, regions=None):
#     if not validation:
#         tr_transforms = []
#
#         # don't do color augmentations while in 2d mode with 3d data because the color channel is overloaded!!
#         if params.get("dummy_2D"):
#             tr_transforms.append(Convert3DTo2DTransform())
#             patch_size_spatial = params.get("patch_size")[1:]
#         else:
#             patch_size_spatial = params.get("patch_size")
#
#         elastic_deform_alpha = (0., 200.)
#         elastic_deform_sigma = (9., 13.)
#         p_eldef = 0.2
#         scale_range = (0.85, 1.25)
#         independent_scale_factor_for_each_axis = False
#         p_scale = 0.2
#         # rotation_x = (-0.349066, 0.349066)
#         # rotation_y = (-0.349066, 0.349066)
#         # rotation_z = (-0.349066, 0.349066)
#         rotation_x = (-0.174533, 0.174533)
#         rotation_y = (-0.174533, 0.174533)
#         rotation_z = (-0.174533, 0.174533)
#         p_rot = 0.2
#         gamma_retain_stats = True
#         gamma_range = (0.7, 1.5)
#         p_gamma = 0.15
#         mirror_axes = (2,)
#         border_mode_data = "constant"
#
#         tr_transforms.append(SpatialTransform(
#             patch_size_spatial, do_elastic_deform=params.get("elastic"), alpha=elastic_deform_alpha, sigma=elastic_deform_sigma,
#             do_rotation=params.get("rotation"), angle_x=rotation_x, angle_y=rotation_y, angle_z=rotation_z,
#             do_scale=params.get("scaling"), scale=scale_range, border_mode_data=border_mode_data,
#             border_cval_data=0, order_data=3, border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=1,
#             random_crop=False, p_el_per_sample=p_eldef, p_scale_per_sample=p_scale,
#             p_rot_per_sample=p_rot, independent_scale_for_each_axis=independent_scale_factor_for_each_axis
#         ))
#         if params.get("dummy_2D"):
#             tr_transforms.append(Convert2DTo3DTransform())
#
#         if params.get("resize_shape"):
#             tr_transforms.append(ResizeTransform(params.get("resize_shape")))
#
#         if params.get("gaussian_noise"):
#             tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.15))
#
#         if params.get("_gaussian_blur"):
#             tr_transforms.append(GaussianBlurTransform((0.5, 1.5), different_sigma_per_channel=True, p_per_sample=0.1,
#                                                        p_per_channel=0.5))
#         if params.get("brightness"):
#             tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15))
#
#         if params.get("contrast"):
#             tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15))
#
#         if params.get("gamma"):
#             tr_transforms.append(
#                 GammaTransform(gamma_range, False, True, retain_stats=gamma_retain_stats,
#                                p_per_sample=p_gamma))
#
#         if params.get("mirror"):
#             tr_transforms.append(MirrorTransform(mirror_axes))
#
#         # tr_transforms.append(RemoveLabelTransform(-1, 0))
#
#         # tr_transforms.append(RenameTransform('seg', 'target', True))
#
#         if regions is not None:
#             tr_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))
#
#         tr_transforms = Compose(tr_transforms)
#
#         return tr_transforms
#     else:
#         # val_transforms = [RemoveLabelTransform(-1, 0)]
#         val_transforms = [SpatialTransform(
#             params.get("patch_size"), do_elastic_deform=False, do_rotation=False, do_scale=False,
#             border_cval_seg=border_val_seg, order_seg=1, random_crop=False
#         )]
#
#         if params.get("resize_shape") is not None:
#             val_transforms.append(ResizeTransform(params.get("resize_shape")))
#
#         # val_transforms.append(RenameTransform('seg', 'target', True))
#
#         if regions is not None:
#             val_transforms.append(ConvertSegmentationToRegionsTransform(regions, 'target', 'target'))
#
#         val_transforms = Compose(val_transforms)
#
#         return val_transforms


def define_nnunet_transformations(params, validation=False):

    transforms = []
    if not validation:

        p_rotation = 0.2 if params['rotation'] else 0
        rotation = params['rot_for_da'] if params['rotation'] else None
        p_scaling = 0.2 if params['scaling'] else 0
        scaling = params['scaling_range'] if params['scaling'] else None
        p_synchronize_scaling_across_axes = 1 if params['scaling'] else None

        if params['dummy_2d']:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = params['patch_size'][1:]
        else:
            patch_size_spatial = params['patch_size']
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=p_rotation,
                rotation=rotation, p_scaling=p_scaling, scaling=scaling,
                p_synchronize_scaling_across_axes=p_synchronize_scaling_across_axes,
            )
        )

        if params['dummy_2d']:
            transforms.append(Convert2DTo3DTransform())

        if params['gaussian_noise']:
            transforms.append(RandomTransform(
                GaussianNoiseTransform(
                    noise_variance=(0, 0.1),
                    p_per_channel=1,
                    synchronize_channels=True
                ), apply_probability=0.1
            ))
        if params['gaussian_blur']:
            transforms.append(RandomTransform(
                GaussianBlurTransform(
                    blur_sigma=(0.5, 1.),
                    synchronize_channels=False,
                    synchronize_axes=False,
                    p_per_channel=0.5, benchmark=True
                ), apply_probability=0.2
            ))
        if params['brightness']:
            transforms.append(RandomTransform(
                MultiplicativeBrightnessTransform(
                    multiplier_range=BGContrast(params['brightness_range']),
                    synchronize_channels=False,
                    p_per_channel=1
                ), apply_probability=0.15
            ))
        if params['contrast']:
            transforms.append(RandomTransform(
                ContrastTransform(
                    contrast_range=BGContrast(params['contrast_range']),
                    preserve_range=True,
                    synchronize_channels=False,
                    p_per_channel=1
                ), apply_probability=0.15
            ))
        if params['low_resolution']:
            transforms.append(RandomTransform(
                SimulateLowResolutionTransform(
                    scale=(0.5, 1),
                    synchronize_channels=False,
                    synchronize_axes=True,
                    ignore_axes=ignore_axes,
                    allowed_channels=None,
                    p_per_channel=0.5
                ), apply_probability=0.25
            ))
        if params['gamma']:
            transforms.append(RandomTransform(
                GammaTransform(
                    gamma=BGContrast(params['gamma_range']),
                    p_invert_image=1,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1
                ), apply_probability=0.
            ))
            transforms.append(RandomTransform(
                GammaTransform(
                    gamma=BGContrast(params['gamma_range']),
                    p_invert_image=0,
                    synchronize_channels=False,
                    p_per_channel=1,
                    p_retain_stats=1
                ), apply_probability=0.3
            ))

        if params['mirror_axes'] is not None and len(params['mirror_axes']) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=params['mirror_axes']
                )
            )

    else:

        transforms.append(
            SpatialTransform(
                params['patch_size'], patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0, p_scaling=0
            )
        )

    return ComposeTransforms(transforms)
