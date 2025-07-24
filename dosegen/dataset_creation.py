import os
import glob
import cv2
import pydicom
import argparse
import trimesh
import warnings
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy.spatial import cKDTree
from skimage.morphology import disk
from functools import partial
from multiprocessing import Pool
from typing import Tuple
from batchgenerators.utilities.file_and_folder_operations import subfiles, save_json
from dcmrtstruct2nii.adapters.convert.rtstructcontour2mask import DcmPatientCoords2Mask
from dcmrtstruct2nii.adapters.input.image.dcminputadapter import DcmInputAdapter
from dcmrtstruct2nii.adapters.input.contours.rtstructinputadapter import RtStructInputAdapter
from dcmrtstruct2nii.exceptions import ContourOutOfBoundsException

from applicator_extractor.applicator import ApplicatorParser


def get_identifiers_from_splitted_files(folder: str):
    uniques = np.unique([i[:-7] for i in subfiles(folder, suffix='.nii.gz', join=False)])
    return uniques


def generate_dataset_json(output_file: str, imagesTr_dir: str, imagesTs_dir: str, modalities: Tuple,
                          labels: dict, dataset_name: str, license: str = "hands off!", dataset_description: str = "",
                          dataset_reference="", dataset_release='0.0'):
    """
    :param output_file: This needs to be the full path to the dataset.json you intend to write, so
    output_file='DATASET_PATH/dataset.json' where the folder DATASET_PATH points to is the one with the
    imagesTr and labelsTr subfolders
    :param imagesTr_dir: path to the imagesTr folder of that dataset
    :param imagesTs_dir: path to the imagesTs folder of that dataset. Can be None
    :param modalities: tuple of strings with modality names. must be in the same order as the images (first entry
    corresponds to _0000.nii.gz, etc). Example: ('T1', 'T2', 'FLAIR').
    :param labels: dict with int->str (key->value) mapping the label IDs to label names. Note that 0 is always
    supposed to be background! Example: {0: 'background', 1: 'edema', 2: 'enhancing tumor'}
    :param dataset_name: The name of the dataset. Can be anything you want
    :param license:
    :param dataset_description:
    :param dataset_reference: website of the dataset, if available
    :param dataset_release:
    :return:
    """
    train_identifiers = get_identifiers_from_splitted_files(imagesTr_dir)

    if imagesTs_dir is not None:
        test_identifiers = get_identifiers_from_splitted_files(imagesTs_dir)
    else:
        test_identifiers = []

    json_dict = {}
    json_dict['name'] = dataset_name
    json_dict['description'] = dataset_description
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = dataset_reference
    json_dict['licence'] = license
    json_dict['release'] = dataset_release
    json_dict['modality'] = {str(i): modalities[i] for i in range(len(modalities))}
    json_dict['labels'] = {str(i): labels[i] for i in labels.keys()}

    json_dict['numTraining'] = len(train_identifiers)
    json_dict['numTest'] = len(test_identifiers)
    json_dict['training'] = [
        {'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i
        in
        train_identifiers]
    json_dict['test'] = ["./imagesTs/%s.nii.gz" % i for i in test_identifiers]

    if not output_file.endswith("dataset.json"):
        print("WARNING: output file name is not dataset.json! This may be intentional or not. You decide. "
              "Proceeding anyways...")
    save_json(json_dict, os.path.join(output_file))


def get_patient_data_from_csv(patient_csv, read_path):
    patient_data = pd.read_csv(patient_csv)

    # Check if all ids in the csv exist in the read path
    all_paths = glob.glob(read_path + '*/')
    all_read_ids = [patient_path.split('/')[-2] for patient_path in all_paths]
    ids_excluded = [patient_id for patient_id in patient_data['patient id'].unique() if patient_id not in all_read_ids]
    assert ids_excluded == [], f"There are patients missing in the dataset: {ids_excluded}"

    for i, row in patient_data.iterrows():
        # correct folder paths with the current read folder
        main_path = read_path + "/".join(row['folder path'].split('\\')[-3:-1]) + '/'
        patient_data.loc[i, 'folder path'] = main_path

    return patient_data


def verify_img_rtstruct(image_path, rtstruct_path):
    if image_path and rtstruct_path:
        slice_ids = []
        mri_slices_paths = glob.glob(image_path + '*')
        for slice_path in mri_slices_paths:
            mri_slice = pydicom.dcmread(slice_path)
            slice_ids.append(mri_slice['SOPInstanceUID'].value)
        rtstruct_file = glob.glob(rtstruct_path + '*')
        assert len(rtstruct_file) == 1, f"{rtstruct_file}"
        rtstruct = pydicom.dcmread(rtstruct_file[0])
        all_reference_ids = []
        rtstruct_sequences = rtstruct['ROIContourSequence'].value
        for item1 in rtstruct_sequences:
            contour_sequence = item1['ContourSequence'].value
            for item2 in contour_sequence:
                contour_item = item2['ContourImageSequence'].value
                assert len(contour_item) == 1
                contour_item = contour_item[0]['ReferencedSOPInstanceUID'].value
                all_reference_ids.append(contour_item)
        unique_ref_ids = list(np.unique(all_reference_ids))
        try:
            assert all([item in slice_ids for item in unique_ref_ids]), f"{sorted(slice_ids)} \n {sorted(unique_ref_ids)}"
            print("    MRI and RTSTRUCT match!")
        except AssertionError:
            warnings.warn(f"MRI {image_path} and RTSTRUCT {rtstruct_path} don't match!")


def get_item_paths(folder_path):
    mr_paths = glob.glob(folder_path + 'MR*/')
    struct_paths = glob.glob(folder_path + 'RTSTRUCT*/')
    dose_paths = glob.glob(folder_path + 'RTDOSE*/')
    plan_paths = glob.glob(folder_path + 'RTPLAN*/')
    app_paths = glob.glob(folder_path + 'applicator*/')

    if len(mr_paths) > 4: warnings.warn(f"Redundant MRIs found: {folder_path.split('/')[-3]}")
    if len(struct_paths) != 1: warnings.warn(f"Zero or more than 1 RTSTRUCTs found: {folder_path.split('/')[-3]}")
    if len(dose_paths) != 1: warnings.warn(f"Zero or more than 1 RTDOSEs found: {folder_path.split('/')[-3]}")
    if len(plan_paths) != 1: warnings.warn(f"Zero or more than 1 RTPLANs found: {folder_path.split('/')[-3]}")
    if len(app_paths) != 1: warnings.warn(f"Zero or more than 1 applicators found: {folder_path.split('/')[-3]}")

    t2_path, bffe_path = None, None
    for mr_path in mr_paths:
        mri_slice_path = glob.glob(mr_path + '*.dcm')[0]
        mri_slice = pydicom.dcmread(mri_slice_path)
        mri_description = mri_slice['SeriesDescription'].value
        if mri_description in ['T2 TSE', 'T2 TRA']:
            t2_path = mr_path
        elif mri_description in ['3D BFFE', '3D bFFE TRA']:
            bffe_path = mr_path

    if struct_paths:
        rtstruct_folder = struct_paths[0]
        rtstruct_path = glob.glob(rtstruct_folder + '*.dcm')
        assert len(rtstruct_path) == 1
        rtstruct_path = rtstruct_path[0]
    else:
        rtstruct_path = None

    if t2_path:
        verify_img_rtstruct(t2_path, rtstruct_path)

    if dose_paths:
        rtdose_folder = dose_paths[0]
        rtdose_path = glob.glob(rtdose_folder + '*.dcm')
        assert len(rtdose_path) == 1
        rtdose_path = rtdose_path[0]
    else:
        rtdose_path = None

    if plan_paths:
        rtplan_folder = plan_paths[0]
        rtplan_path = glob.glob(rtplan_folder + '*.dcm')
        assert len(rtplan_path) == 1
        rtplan_path = rtplan_path[0]
    else:
        rtplan_path = None

    if app_paths:
        app_folder = app_paths[0]
        app_path = glob.glob(app_folder + '*.dcm')
        assert len(app_path) == 1
        app_path = app_path[0]
    else:
        app_path = None

    item_paths = {'t2': t2_path, 'bffe': bffe_path, 'struct': rtstruct_path,
                  'dose': rtdose_path, 'plan': rtplan_path, 'app': app_path}

    return item_paths


def generate_line_points(line_points, mode='dist', n_points=None, dist=None, eps=1e-10):
    """
    Generate equidistant points along a polyline using linear interpolation.
    Ensures the first and last points remain unchanged.

    Parameters:
    - line_points: (n, d) array representing the input polyline points.
    - mode: 'dist' for a fixed step size, 'n_points' for a fixed number of points.
    - n_points: Number of points to generate (if mode='n_points').
    - dist: Distance between sampled points (if mode='dist').
    - eps: Small epsilon to avoid numerical precision issues.

    Returns:
    - resampled points as an array of shape (m, d).
    """
    line_points = np.asarray(line_points)
    lengths = np.sqrt(np.sum(np.square(line_points[1:, :] - line_points[:-1, :]), axis=1))
    total_length = np.sum(lengths)

    if mode == 'n_points':
        dist = total_length / (n_points - 1 + eps)

    max_line_coords = np.cumsum(np.concatenate([[0], lengths])) / dist

    def get_line_interps(p0, p1, c0, c1, l, dist):
        """Generate interpolated points between two given points."""
        p0 = p0.reshape(1, -1)
        p1 = p1.reshape(1, -1)
        v = p1 - p0  # Direction vector
        n = v / l  # Normalized direction
        b0 = np.ceil(c0)
        b1 = np.floor(c1)
        c = np.arange(b0, b1 + 1) - c0
        return p0 + dist * n * c.reshape(-1, 1)

    # Generate interpolated points
    result = np.concatenate([
        get_line_interps(p0, p1, c0, c1, l, dist)
        for (p0, p1, c0, c1, l) in zip(line_points, line_points[1:], max_line_coords, max_line_coords[1:], lengths)
    ])

    # Ensure first and last points are exactly the same as input
    result[0] = line_points[0]
    result[-1] = line_points[-1]

    return result


def check_needle_applicator_intersection_anisotropic(line, mesh, thresholds=(1.1, 1.1, 0.1), n_sampled_points=200):
    """
    Checks if any point in the given prediction line is within anisotropic thresholds
    from the mesh surface.

    Parameters:
        mesh (trimesh.Trimesh): The mesh object.
        line (np.ndarray): A (K, 3) numpy array representing a line with K points in 3D space.
        thresholds (tuple): A (3,) tuple for x, y, z thresholds.

    Returns:
        bool: True if any point in the line is within anisotropic threshold of the mesh.
    """

    # First sample points cause the line consists of only 2 points
    sampled_line = generate_line_points(line, mode='n_points', n_points=n_sampled_points)
    # Don't use the top half points
    sampled_line = sampled_line[sampled_line.shape[0] // 2:]

    # Scale coordinates according to thresholds
    scaling = np.array([1.0 / thresholds[0], 1.0 / thresholds[1], 1.0 / thresholds[2]])

    scaled_line = sampled_line * scaling

    scaled_vertices = mesh.vertices * scaling

    # Build KDTree on mesh vertices (fast)
    kdtree = cKDTree(scaled_vertices)

    # Query distances
    distances, _ = kdtree.query(scaled_line, k=1)

    # Use isotropic threshold of 1.0 in scaled space
    return np.any(distances < 1.0)


def extend_needle_bottom_points(rtplan, needle_points):
    applicator = ApplicatorParser(rtplan).parse()
    # applicator_mesh = applicator.get_transformed_skin()
    full_model = applicator.get_combined_skin()
    vertices = full_model.vertices  # for Vangelis
    faces = full_model.faces  # for Vangelis

    mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

    for i, line in enumerate(needle_points):
        top_point, bot_point = line
        direction = bot_point - top_point
        direction = 0.5 * (direction / np.linalg.norm(direction))
        max_steps = 50

        while max_steps > 0:
            # Find matches: same x and y, higher z
            overlap = check_needle_applicator_intersection_anisotropic(line, mesh)
            # move bot_point towards the applicator surface
            if overlap:
                bot_point = bot_point - direction
                line = np.stack([top_point, bot_point])
                new_overlap = check_needle_applicator_intersection_anisotropic(line, mesh)
                if not new_overlap:
                    bot_point = bot_point + direction
                    break
            else:
                bot_point = bot_point + direction
                line = np.stack([top_point, bot_point])
                new_overlap = check_needle_applicator_intersection_anisotropic(line, mesh)
                if new_overlap:
                    # bot_point = bot_point - direction
                    break

            max_steps -= 1

        if max_steps == 0:
            print('    Free needle detected. Skipping extension.')
        else:
            needle_points[i][-1] = bot_point

    return needle_points


def dilation_per_slice(image, radius):
    kernel = disk(radius)  # Proper circular structuring element
    image_slices = [cv2.dilate(image_slice.astype('uint8'), kernel) for image_slice in image]
    return np.stack(image_slices, axis=0)


def extract_manual_needle_points(plan_file_path, return_name_dict=False):
    rtplan = pydicom.dcmread(plan_file_path)
    needle_labels = ['naald', 'nld', 'nald', 'Naald', 'Nld', 'Nald', 'NAALD', 'NLD', 'NALD']
    structureSetROISequence = rtplan[0x300f, 0x1000].value[0][0x3006, 0x0020].value
    ids_and_names_list = [(item[0x3006, 0x0022].value, item[0x3006, 0x0026].value) for item in structureSetROISequence]
    needle_name_dict = {item[0]: item[1] for item in ids_and_names_list if
                        any([lbl in item[1] for lbl in needle_labels])}
    rOIContourSequence = rtplan[0x300f, 0x1000].value[0][0x3006, 0x0039].value

    manual_needle_points = {}
    for item in rOIContourSequence:
        if item[0x3006, 0x0084].value in needle_name_dict.keys():
            # get manually defined needle points
            temp_points = item[0x3006, 0x0040].value[0][0x3006, 0x0050].value
            temp_points = np.array([temp_points[i:i + 3] for i in range(0, len(temp_points), 3)])
            temp_points = np.array(sorted(temp_points, key=lambda x: x[-1], reverse=True))
            # keep only the top 2 points
            manual_needle_points[item[0x3006, 0x0084].value] = temp_points[:2]

    manual_needle_points = list(manual_needle_points.values())
    if manual_needle_points:
        manual_needle_points = extend_needle_bottom_points(rtplan, manual_needle_points)

    if return_name_dict:
        return manual_needle_points, needle_name_dict
    else:
        return manual_needle_points


def create_needle_contours(rtplan_file_path, sitk_image, radius):
    manual_points, needle_name_dict = extract_manual_needle_points(rtplan_file_path, return_name_dict=True)
    needle_mask_np = np.zeros_like(sitk.GetArrayFromImage(sitk_image)).astype('uint8')

    if needle_name_dict:
        for needle_points in manual_points:
            interp_points = generate_line_points(needle_points, dist=0.1)
            for p in interp_points:
                idx = sitk_image.TransformPhysicalPointToIndex(p.tolist())
                needle_mask_np[tuple(reversed(idx))] = 1

        needle_mask_np = dilation_per_slice(needle_mask_np, radius=radius)

    needle_mask = sitk.GetImageFromArray(needle_mask_np.astype('uint8'))
    needle_mask.CopyInformation(sitk_image)
    return needle_mask


def create_applicator_contours(rtplan_file_path, sitk_image, radius):
    rtplan = pydicom.dcmread(rtplan_file_path)
    sequences = rtplan['ApplicationSetupSequence'].value[0][0x300a, 0x0280].value
    applicator_mask_np = np.zeros_like(sitk.GetArrayFromImage(sitk_image)).astype('uint8')

    for i in range(len(sequences)):
        temp_sequence = sequences[i][0x300a, 0x02d0].value
        if i in [0, 1, 2]:
            temp_points = []
            for j in range(len(temp_sequence)):
                point_location = temp_sequence[j][0x300a, 0x02d4].value
                temp_points.append(point_location)
            _, indices = np.unique(temp_points, axis=0, return_index=True)
            dwell_locations = np.array(temp_points)[np.sort(indices)]
            interp_points = generate_line_points(dwell_locations, dist=0.1)
            for p in interp_points:
                idx = sitk_image.TransformPhysicalPointToIndex(p.tolist())
                applicator_mask_np[tuple(reversed(idx))] = 1

    applicator_mask_np = dilation_per_slice(applicator_mask_np, radius=radius)

    applicator_mask = sitk.GetImageFromArray(applicator_mask_np.astype('uint8'))
    applicator_mask.CopyInformation(sitk_image)
    return applicator_mask


def extract_rtstructs(dicom_image, rtstruct_file_path, target_labels, mask_background_value=0, mask_foreground_value=1):
    rtreader = RtStructInputAdapter()
    rtstructs = rtreader.ingest(rtstruct_file_path)
    dcm_patient_coords_to_mask = DcmPatientCoords2Mask()
    dummy_zeros_mask_np = np.zeros_like(sitk.GetArrayFromImage(dicom_image)).astype('uint8')
    class_masks = {}

    for label_idx in target_labels:
        current_rtstruct = [rtstruct for rtstruct in rtstructs if rtstruct['name'] == target_labels[label_idx]]
        if current_rtstruct:
            assert len(current_rtstruct) == 1
            if 'sequence' not in current_rtstruct[0]:
                warnings.warn(f"Mask for {current_rtstruct[0]['name']} will be empty. No shape/polygon found.")
                mask = sitk.GetImageFromArray(dummy_zeros_mask_np)
            else:
                try:
                    mask = dcm_patient_coords_to_mask.convert(current_rtstruct[0]['sequence'], dicom_image,
                                                              mask_background_value, mask_foreground_value)
                except ContourOutOfBoundsException:
                    warnings.warn(f"Structure {current_rtstruct[0]['name']} is out of bounds, ignoring contour!")
                    mask = sitk.GetImageFromArray(dummy_zeros_mask_np)
        else:
            if target_labels[label_idx] != 'background':
                warnings.warn(f'Mask for {target_labels[label_idx]} will be empty.')
            mask = sitk.GetImageFromArray(dummy_zeros_mask_np)

        mask.CopyInformation(dicom_image)
        class_masks[label_idx] = mask
    return class_masks


def extract_sitk_masks_from_dicoms(t2, bffe, target_labels, rtstruct_path, rtplan_path):

    class_masks_1 = extract_rtstructs(t2, rtstruct_path,
                                      target_labels={key: value for key, value in target_labels.items()
                                                     if value not in ['applicator', 'needles']})

    dummy_zeros_mask_np = np.zeros_like(sitk.GetArrayFromImage(bffe)).astype('uint8')
    dummy_zeros_mask = sitk.GetImageFromArray(dummy_zeros_mask_np)
    dummy_zeros_mask.CopyInformation(bffe)
    applicator_idx = list(target_labels.keys())[list(target_labels.values()).index('applicator')]
    needle_idx = list(target_labels.keys())[list(target_labels.values()).index('needles')]
    class_masks_2 = {"0": dummy_zeros_mask, applicator_idx: create_applicator_contours(rtplan_path, bffe, 6),
                     needle_idx: create_needle_contours(rtplan_path, bffe, 4)}

    # --- Combine class masks 1 (T2) ---
    combined_np_1 = np.zeros_like(sitk.GetArrayFromImage(t2), dtype=np.uint8)
    for class_idx_str, sitk_mask in class_masks_1.items():
        class_idx = int(class_idx_str)
        mask_np = sitk.GetArrayFromImage(sitk_mask)
        combined_np_1[mask_np > 0] = class_idx
    combined_mask_1 = sitk.GetImageFromArray(combined_np_1)
    combined_mask_1.CopyInformation(t2)

    # --- Combine class masks 2 (BFFE) ---
    combined_np_2 = np.zeros_like(sitk.GetArrayFromImage(bffe), dtype=np.uint8)
    for class_idx_str, sitk_mask in class_masks_2.items():
        class_idx = int(class_idx_str)
        mask_np = sitk.GetArrayFromImage(sitk_mask)
        combined_np_2[mask_np > 0] = class_idx
    combined_mask_2 = sitk.GetImageFromArray(combined_np_2)
    combined_mask_2.CopyInformation(bffe)

    return combined_mask_1, combined_mask_2


def get_physical_corners(img):
    size = np.array(img.GetSize())
    spacing = np.array(img.GetSpacing())
    direction = np.array(img.GetDirection()).reshape(3, 3)
    origin = np.array(img.GetOrigin())

    # Compute corner points
    index_corners = np.array(np.meshgrid(
        [0, size[0]],
        [0, size[1]],
        [0, size[2]],
    )).T.reshape(-1, 3)

    # Physical points
    physical_corners = [origin + direction @ (spacing * idx) for idx in index_corners]
    return np.min(physical_corners, axis=0), np.max(physical_corners, axis=0)


def clamp_index(index, image):
    size = image.GetSize()
    return [max(0, min(i, size[d] - 1)) for d, i in enumerate(index)]


def crop_image_to_physical_bounds(image, phys_min, phys_max):
    # Convert physical to index coordinates
    index_min = clamp_index(image.TransformPhysicalPointToIndex(phys_min), image)
    index_max = clamp_index(image.TransformPhysicalPointToIndex(phys_max), image)

    size = [e - s for s, e in zip(index_min, index_max)]

    # Crop the image
    roi_filter = sitk.RegionOfInterestImageFilter()
    roi_filter.SetIndex(index_min)
    roi_filter.SetSize(size)
    return roi_filter.Execute(image)


def resample_to_reference(image: sitk.Image, reference: sitk.Image, is_mask=False) -> sitk.Image:
    if is_mask:
        return resample_mask_to_reference(image, reference)
    else:
        resample = sitk.ResampleImageFilter()
        resample.SetReferenceImage(reference)
        resample.SetOutputSpacing(reference.GetSpacing())
        resample.SetSize(reference.GetSize())
        resample.SetOutputOrigin(reference.GetOrigin())
        resample.SetOutputDirection(reference.GetDirection())
        resample.SetDefaultPixelValue(0)
        resample.SetTransform(sitk.Transform())
        resample.SetInterpolator(sitk.sitkBSpline)
        return resample.Execute(image)


def resample_mask_to_reference(mask_image: sitk.Image, reference: sitk.Image) -> sitk.Image:
    mask_array = sitk.GetArrayFromImage(mask_image)
    labels = np.unique(mask_array)
    one_hot = np.stack([(mask_array == label).astype(np.float32) for label in labels], axis=0)
    resampled_channels = []

    for ch in one_hot:
        ch_img = sitk.GetImageFromArray(ch)
        ch_img.CopyInformation(mask_image)

        # Set up full 3D resampler
        resampler = sitk.ResampleImageFilter()
        resampler.SetReferenceImage(reference)
        resampler.SetSize(reference.GetSize())
        resampler.SetOutputSpacing(reference.GetSpacing())
        resampler.SetOutputOrigin(reference.GetOrigin())
        resampler.SetOutputDirection(reference.GetDirection())
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetTransform(sitk.Transform())
        resampler.SetDefaultPixelValue(0.0)

        resampled = resampler.Execute(ch_img)
        resampled_channels.append(sitk.GetArrayFromImage(resampled))

    # Stack and argmax over channel axis
    stacked = np.stack(resampled_channels, axis=0)
    argmax_output = np.argmax(stacked, axis=0)

    # Map back to original labels
    output_array = np.zeros_like(argmax_output, dtype=np.uint8)
    for idx, label in enumerate(labels):
        output_array[argmax_output == idx] = label

    # Convert to SimpleITK image
    output_img = sitk.GetImageFromArray(output_array)
    output_img.CopyInformation(reference)
    return output_img


def resample_to_match_direction(image, reference_image):
    if np.allclose(image.GetDirection(), reference_image.GetDirection(), atol=1e-6):
        return image  # Already aligned

    # Get direction info
    original_direction = np.array(image.GetDirection()).reshape(3, 3)
    target_direction = np.array(reference_image.GetDirection()).reshape(3, 3)

    # Compute center of original image in physical space
    size = np.array(image.GetSize())
    spacing = np.array(image.GetSpacing())
    origin = np.array(image.GetOrigin())

    center_index = size / 2.0
    center_physical = origin + original_direction @ (spacing * center_index)

    # Compute new origin to keep the center fixed after rotation
    new_origin = center_physical - target_direction @ (spacing * center_index)

    # Set up resampling
    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(image.GetSize())
    resampler.SetOutputSpacing(spacing.tolist())
    resampler.SetOutputOrigin(new_origin.tolist())
    resampler.SetOutputDirection(reference_image.GetDirection())
    resampler.SetTransform(sitk.Transform())  # Identity

    if image.GetPixelID() in (sitk.sitkUInt8, sitk.sitkInt8):
        resampler.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resampler.SetInterpolator(sitk.sitkLinear)

    resampler.SetDefaultPixelValue(0)
    return resampler.Execute(image)


def combine_label_masks(oar_mask, app_mask, target_labels):
    oar_mask_np = sitk.GetArrayFromImage(oar_mask)
    app_mask_np = sitk.GetArrayFromImage(app_mask)

    if oar_mask_np.shape != app_mask_np.shape:
        raise ValueError(
            f"OAR and applicator masks must have the same shape: {oar_mask_np.shape} and {app_mask_np.shape}")

    combined_mask = np.zeros_like(oar_mask_np, dtype=np.uint8)

    # Convert keys once for faster lookup
    valid_labels = set(int(k) for k in target_labels.keys())

    # Merge OAR mask
    for label in np.unique(oar_mask_np):
        if label == 0:
            continue
        if label not in valid_labels:
            raise ValueError(f"OAR mask contains unknown label {label}")
        combined_mask[oar_mask_np == label] = label

    # Merge applicator mask
    for label in np.unique(app_mask_np):
        if label == 0:
            continue
        if label not in valid_labels:
            raise ValueError(f"Applicator mask contains unknown label {label}")
        if np.any((combined_mask > 0) & (app_mask_np == label)):
            warnings.warn(f"Label {label} overlaps between OAR and applicator masks. Giving priority to OARs.")
        combined_mask[(app_mask_np == label) & ~(combined_mask > 0)] = label

    mask = sitk.GetImageFromArray(combined_mask)
    mask.CopyInformation(oar_mask)

    return mask


def overlay_images_with_rtdose(t2, bffe, rtdose, **kwargs):
    oar_mask, app_mask = extract_sitk_masks_from_dicoms(t2, bffe, **kwargs)

    def sp_to_str(spacing):
        return tuple(float(f'{item:.2f}') for item in spacing)

    print('    Initial information of current patient images:')
    print(f'        T2: {t2.GetSize()}, {sp_to_str(t2.GetSpacing())}, {sp_to_str(t2.GetOrigin())}, {t2.GetDirection()}')
    print(f'        BFFE: {bffe.GetSize()}, {sp_to_str(bffe.GetSpacing())}, {sp_to_str(bffe.GetOrigin())}, {bffe.GetDirection()}')
    print(f'        OAR_MASK: {oar_mask.GetSize()}, {sp_to_str(oar_mask.GetSpacing())}, {sp_to_str(oar_mask.GetOrigin())}, {oar_mask.GetDirection()}')
    print(f'        APP_MASK: {app_mask.GetSize()}, {sp_to_str(app_mask.GetSpacing())}, {sp_to_str(app_mask.GetOrigin())}, {app_mask.GetDirection()}')
    print(f'        DOSE: {rtdose.GetSize()}, {sp_to_str(rtdose.GetSpacing())}, {sp_to_str(rtdose.GetOrigin())}, {rtdose.GetDirection()}')

    print("    Restricting field of view and resampling everything based on the dose...")
    t2 = resample_to_match_direction(t2, rtdose)
    bffe = resample_to_match_direction(bffe, rtdose)
    oar_mask = resample_to_match_direction(oar_mask, rtdose)
    app_mask = resample_to_match_direction(app_mask, rtdose)

    t2_min, t2_max = get_physical_corners(t2)
    bffe_min, bffe_max = get_physical_corners(bffe)
    dose_min, dose_max = get_physical_corners(rtdose)

    # Select the smallest bounding box
    main_origin = np.max([t2_min, bffe_min, dose_min], axis=0)
    main_end = np.min([t2_max, bffe_max, dose_max], axis=0)

    t2 = crop_image_to_physical_bounds(t2, main_origin, main_end)
    bffe = crop_image_to_physical_bounds(bffe, main_origin, main_end)
    rtdose = crop_image_to_physical_bounds(rtdose, main_origin, main_end)
    oar_mask = crop_image_to_physical_bounds(oar_mask, main_origin, main_end)
    app_mask = crop_image_to_physical_bounds(app_mask, main_origin, main_end)

    t2 = resample_to_reference(t2, rtdose, is_mask=False)
    bffe = resample_to_reference(bffe, rtdose, is_mask=False)
    oar_mask = resample_to_reference(oar_mask, rtdose, is_mask=True)
    app_mask = resample_to_reference(app_mask, rtdose, is_mask=True)

    image = sitk.JoinSeries([t2, bffe])
    mask = combine_label_masks(oar_mask, app_mask, kwargs['target_labels'])

    print('    Final information:')
    print(f'        size: {rtdose.GetSize()}')
    print(f'        spacing: {rtdose.GetSpacing()}')
    print(f'        origin: {rtdose.GetOrigin()}')
    print(f'        direction: {rtdose.GetDirection()}')

    return image, rtdose, mask


def process_and_save_patient(patient_index, patient_row_data, target_labels, dataset_id, images_path, labels_path, doses_path):
    print(f"Processing data from patient {patient_row_data['patient id']}...")

    path_to_save_img_nifti_file = images_path + dataset_id.lower() + f"_{patient_index:0=3d}.nii.gz"
    path_to_save_mask_nifti_file = labels_path + dataset_id.lower() + f"_{patient_index:0=3d}.nii.gz"
    path_to_save_dose_nifti_file = doses_path + dataset_id.lower() + f"_{patient_index:0=3d}.nii.gz"

    item_paths = get_item_paths(patient_row_data['folder path'])
    kwargs = {'target_labels': target_labels, 'rtstruct_path': item_paths['struct'], 'rtplan_path': item_paths['plan']}

    print(f"    T2 series path: {item_paths['t2']}")
    print(f"    BFFE series path: {item_paths['bffe']}")
    print(f"    RTSTRUCT file path: {item_paths['struct']}")
    print(f"    RTPLAN file path: {item_paths['plan']}")

    t2 = DcmInputAdapter().ingest(item_paths['t2'])
    bffe = DcmInputAdapter().ingest(item_paths['bffe'])
    rtdose = sitk.ReadImage(item_paths['dose'])

    image, rtdose, mask = overlay_images_with_rtdose(t2, bffe, rtdose, **kwargs)

    # write image and mask to nii files
    print(f'    Writing image to {path_to_save_img_nifti_file}...')
    print(f'    Writing mask to {path_to_save_mask_nifti_file}...')
    print(f'    Writing dose to {path_to_save_dose_nifti_file}...')
    sitk.WriteImage(image, path_to_save_img_nifti_file)
    sitk.WriteImage(mask, path_to_save_mask_nifti_file)
    sitk.WriteImage(rtdose, path_to_save_dose_nifti_file)


def fun_wrapper(dict_args):
    internal_dict = {key: value for key, value in dict_args.items() if key != 'func'}
    dict_args['func'](**internal_dict)


def create_dataset_from_dicoms(data_path, save_path, patient_csv, target_labels, processes):

    assert not os.path.exists(save_path), f"Path '{save_path}' already exists."

    dataset_id = save_path.split('/')[-2].split('_')[-1]

    # create save folders
    images_path = save_path + 'images/'
    labels_path = save_path + 'labels/'
    doses_path = save_path + 'doses/'
    os.mkdir(save_path), os.mkdir(images_path), os.mkdir(labels_path), os.mkdir(doses_path)

    patient_data = get_patient_data_from_csv(patient_csv, data_path)

    partial_func = partial(process_and_save_patient, target_labels=target_labels, dataset_id=dataset_id,
                           images_path=images_path, labels_path=labels_path, doses_path=doses_path)

    pool_args = [{'func': partial_func, 'patient_index': index + 1, 'patient_row_data': patient_row}
                 for index, patient_row in patient_data.iterrows()]
    with Pool(processes) as pool:
        pool.map(fun_wrapper, pool_args)

    generate_dataset_json(output_file=save_path + 'dataset.json', imagesTr_dir=images_path, imagesTs_dir=None,
                          modalities=('T2', 'BFFE'), labels=target_labels, dataset_name=dataset_id,
                          license='', dataset_description='', dataset_reference="", dataset_release='0.0')


def parse_labels(label_str):
    labels = label_str.split(',')
    return {i: label.strip() for i, label in enumerate(labels)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and create dataset.")
    parser.add_argument("dataset_path", type=str, help="Path to raw dataset folder.")
    parser.add_argument("save_path", type=str, help="Path to save processed dataset.")
    parser.add_argument("csv_path", type=str, help="Path to csv file with data paths.")
    parser.add_argument('target_labels', type=parse_labels, help='Comma-separated list of labels.')

    args = parser.parse_args()
    dataset_path = args.dataset_path
    save_path = args.save_path
    csv_path = args.csv_path
    target_labels = args.target_labels

    create_dataset_from_dicoms(dataset_path, save_path, csv_path, target_labels, processes=8)
