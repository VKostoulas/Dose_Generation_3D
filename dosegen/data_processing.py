import os
import torch
import glob
import json
import zarr
import numpy as np

from functools import partial
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import KFold, train_test_split
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

    dataset_path = glob.glob(os.getenv('dosegen_preprocessed') + f'/Task{dataset_id}*/')[0]

    split_file_name = "splits_train_val_test.json" if splitting == "train-val-test" else "splits_final.json"
    split_file_path = os.path.join(dataset_path, split_file_name)

    if os.path.exists(split_file_path):
        print(f"Split file already exists at {split_file_path}. Using this for training.")
        return split_file_path

    file_paths = glob.glob(os.path.join(dataset_path, 'data', "*.zarr"))
    file_names = [os.path.basename(fp).replace('.zarr', '') for fp in file_paths]

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


def smart_crop_from_zarr(zarr_array, bbox, pad_value=0):
    crop_dims = len(bbox)
    img_shape = zarr_array.shape
    num_dims = len(img_shape)

    slices = []
    padding = []
    output_shape = list(img_shape[:num_dims - crop_dims]) + [max_val - min_val for min_val, max_val in bbox]

    for i in range(num_dims):
        if i < num_dims - crop_dims:
            slices.append(slice(None))
            padding.append([0, 0])
        else:
            dim_idx = i - (num_dims - crop_dims)
            min_val, max_val = bbox[dim_idx]

            valid_min = max(min_val, 0)
            valid_max = min(max_val, img_shape[i])
            slices.append(slice(valid_min, valid_max))

            pad_before = max(0, -min_val)
            pad_after = max(0, max_val - img_shape[i])
            padding.append([pad_before, pad_after])

    cropped = zarr_array[tuple(slices)]  # convert only the required crop to memory

    pad_width = [(p[0], p[1]) for p in padding]
    padded = np.pad(cropped, pad_width=pad_width, mode='constant', constant_values=pad_value)

    return padded


class MedicalDataset(Dataset):
    def __init__(self, config, data_ids, batch_size, section, transformation_args, probabilistic_oversampling=False):
        self.data_path = config['data_path']
        self.ids = data_ids
        self.batch_size = batch_size
        self.section = section
        self.transformation_args = transformation_args
        self.oversample_foreground_percent = config['oversample_ratio']
        self.load_mode = config['gen_mode']

        if 'label' in self.load_mode:
            self.n_classes = config['dataset_config']['n_classes']

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
        self.initial_patch_size = (1, *self.initial_patch_size) if len(
            self.initial_patch_size) == 2 else self.initial_patch_size

        self.need_to_pad = (np.array(self.initial_patch_size) - np.array(self.patch_size)).astype(int)
        self.oversampling_method = self._oversample_last_XX_percent if not probabilistic_oversampling \
            else self._probabilistic_oversampling

        validation = False if self.section == "training" else True
        self.transformations = define_nnunet_transformations(self.transformation_args, validation)

    def __len__(self):
        return len(self.ids)

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
                rotation_for_DA = (-np.pi * 15 / 180, np.pi * 15 / 180) if max(self.patch_size) / min(
                    self.patch_size) > 1.5 else (-np.pi, np.pi)
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

    def transform(self, image, dose, label):
        # Compose input image based on load_mode
        inputs = []
        if 'image' in self.load_mode:
            inputs.append(image)
        if 'dose' in self.load_mode:
            inputs.append(dose)
        final_image = torch.cat(inputs, dim=0) if inputs else None

        # Apply transformations
        transform_args = {"image": final_image}
        if 'label' in self.load_mode:
            transform_args["segmentation"] = label
        transformed = self.transformations(**transform_args)

        # Extract transformed outputs
        transformed_label = (torch.squeeze(transformed["segmentation"], dim=0)
                             if 'label' in self.load_mode else None)

        if 'image' in self.load_mode and 'dose' in self.load_mode:
            transformed_image = transformed["image"][:-1]
            transformed_dose = transformed["image"][-1]
        elif 'image' in self.load_mode:
            transformed_image = transformed["image"]
            transformed_dose = None
        elif 'dose' in self.load_mode:
            transformed_image = None
            transformed_dose = transformed["image"]
        else:
            transformed_image = transformed_dose = None

        return transformed_image, transformed_dose, transformed_label

    def load_image_dose_label_properties(self, name):
        zarr_path = os.path.join(self.data_path, name + '.zarr')
        zgroup = zarr.open_group(zarr_path, mode='r')
        image = zgroup['image']
        dose = zgroup['dose']
        label = zgroup['label']
        properties = zgroup.attrs['properties']

        return image, dose, label, properties

    def __getitem__(self, indexes):
        batch_idx, sample_idx = indexes
        name = self.ids[sample_idx]

        image, dose, label, properties = self.load_image_dose_label_properties(name)

        # Decide if oversampling foreground is needed
        force_fg = self.oversampling_method(batch_idx)

        # Get bounding box for cropping
        shape = image.shape[1:]
        bbox_lbs, bbox_ubs = self.get_bbox(shape, force_fg, properties['class_locations'], is_2d=self.patch_size[0] == 1)
        bbox = [[int(i), int(j)] for i, j in zip(bbox_lbs, bbox_ubs)]

        if 'image' in self.load_mode:
            image = smart_crop_from_zarr(image, bbox, 0)
            if len(image.shape) < len(self.patch_size) + 1:
                image = np.expand_dims(image, axis=0)
            image = np.squeeze(image, axis=1) if self.patch_size[0] == 1 else image
            image = torch.as_tensor(image).float().contiguous()
        if 'dose' in self.load_mode:
            dose = smart_crop_from_zarr(dose, bbox, 0)
            if len(dose.shape) < len(self.patch_size) + 1:
                dose = np.expand_dims(dose, axis=0)
            dose = np.squeeze(dose, axis=0) if self.patch_size[0] == 1 else dose
            dose = torch.as_tensor(dose).float().contiguous()
        if 'label' in self.load_mode:
            label = smart_crop_from_zarr(label, bbox, 0)
            if len(label.shape) < len(self.patch_size) + 1:
                label = np.expand_dims(label, axis=0)
            label = np.squeeze(label, axis=0) if self.patch_size[0] == 1 else label
            label = torch.as_tensor(label).to(torch.int16).contiguous()

        # Apply transformations
        image, dose, label = self.transform(image, dose, label)

        outputs = {'id': name}
        input_parts = []

        if 'image' in self.load_mode:
            input_parts.append(image)

        if 'dose' in self.load_mode:
            input_parts.append(dose)

        if 'label' in self.load_mode:
            input_parts.append(label)

        # Concatenate all inputs along channel (0) dimension
        input_tensor = torch.cat(input_parts, dim=0)
        outputs['input'] = input_tensor

        return outputs


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

    train_ds = MedicalDataset(config=config, data_ids=data_ids['train'], batch_size=batch_size,
                              section="training", transformation_args=transformations)
    val_ds = MedicalDataset(config=config, data_ids=data_ids['val'], batch_size=batch_size,
                            section="validation", transformation_args=transformations)

    train_sampler = CustomBatchSampler(train_ds, batch_size=batch_size, number_of_steps=250, shuffle=True)
    val_sampler = CustomBatchSampler(val_ds, batch_size=batch_size, number_of_steps=50, shuffle=False)
    loader_args = dict(num_workers=config['num_workers'], pin_memory=True, prefetch_factor=2)
    train_loader = DataLoader(train_ds, batch_sampler=train_sampler, **loader_args)
    val_loader = DataLoader(val_ds, batch_sampler=val_sampler,**loader_args)
    return train_loader, val_loader






