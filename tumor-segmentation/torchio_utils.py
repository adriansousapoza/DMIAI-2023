import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchio as tio
from pathlib import Path
import shutil
import json
from tqdm import tqdm
import cv2
import gc

# Updated dataset_json_content
dataset_json_content = {
    "channel_names": {
        "0": "MIP-PET"   # This will be updated later
    },
    "labels": {
        "background": 0,
        "TargetRegion": 1
    },
    "numTraining": 0,  # This will be updated later
    "file_ending": ".png"
}

def save_inverted_images(dataset, inverted_images_dir):
    os.makedirs(inverted_images_dir, exist_ok=True)

    for i, subject in enumerate(dataset):
        # Get the already inverted image array
        inverted_image_array = subject.img[tio.DATA].squeeze().numpy()
        inverted_image_array = inverted_image_array.astype(np.uint8)
        inverted_image_pil = Image.fromarray(inverted_image_array[0, :, :])
        inverted_image_filename = f'patient_{i:04d}.png'
        inverted_image_pil.save(os.path.join(inverted_images_dir, inverted_image_filename))

        del inverted_image_array, inverted_image_pil
        gc.collect()

def save_nnUNet_raw_original(dataset, base_dir, dataset_ID, num_subjects, file_ending='.png'):
    # Adjust dataset directory naming convention
    dataset_dir = Path(base_dir) / f"Dataset{int(dataset_ID):03d}"
    images_dir = dataset_dir / 'imagesTr'
    labels_dir = dataset_dir / 'labelsTr'

    # Create directories
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Save images and labels with correct naming convention
    print(f'Saving {num_subjects} subjects to {dataset_dir}...')
    for i, subject in tqdm(enumerate(dataset), total=len(dataset)):
        # Image processing
        image_array = subject.img[tio.DATA].squeeze().numpy()
        image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255.0
        image_array = image_array.astype(np.uint8)
        image_filename = f'patient_{i:04d}_0000{file_ending}'  # Assuming single channel (MIP-PET)
        cv2.imwrite(str(images_dir / image_filename), image_array[0, :, :])

        # Label processing
        label_array = subject.label[tio.DATA].squeeze().numpy()
        binary_label_array = (label_array > 0).astype(np.uint8)  # Convert to binary (0 and 1)
        label_filename = f'patient_{i:04d}{file_ending}'
        cv2.imwrite(str(labels_dir / label_filename), binary_label_array[0, :, :])

        del image_array, label_array, binary_label_array
        gc.collect()

def save_nnUNet_raw_control(dataset, base_dir, dataset_ID, num_subjects, init, file_ending='.png'):
    # Adjust dataset directory naming convention
    dataset_dir = Path(base_dir) / f"Dataset{int(dataset_ID):03d}"
    images_dir = dataset_dir / 'imagesTr'
    labels_dir = dataset_dir / 'labelsTr'

    # Create directories
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    # Save images and labels with correct naming convention
    print(f'Saving {num_subjects} controls to {dataset_dir}...')
    for subject in tqdm(dataset):
        # Image processing
        image_array = subject.img[tio.DATA].squeeze().numpy()
        image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255.0
        image_array = image_array.astype(np.uint8)
        image_filename = f'patient_{init:04d}_0000{file_ending}'  # Assuming single channel (MIP-PET)
        cv2.imwrite(str(images_dir / image_filename), image_array[0, :, :])

        # Label processing
        label_array = subject.label[tio.DATA].squeeze().numpy()
        binary_label_array = (label_array > 0).astype(np.uint8)  # Convert to binary (0 and 1)
        label_filename = f'patient_{init:04d}{file_ending}'
        cv2.imwrite(str(labels_dir / label_filename), binary_label_array[0, :, :])

        init += 1

        del image_array, label_array, binary_label_array
        gc.collect()
    
def create_json_file(dataset_dir, num_subjects, file_ending='.png'):
    # Create dataset.json content
    dataset_json_content = {
        "channel_names": {"0": "MIP-PET"},
        "labels": {"background": 0, "TargetRegion": 1},
        "numTraining": num_subjects,
        "file_ending": file_ending
    }

    # Write dataset.json file
    dataset_json_path = dataset_dir / 'dataset.json'
    with open(dataset_json_path, 'w') as f:
        json.dump(dataset_json_content, f, indent=4)

def save_nnUNet_raw_augmented(subject, base_dir, dataset_ID, index, file_ending='.png'):
    # Adjust dataset directory naming convention
    dataset_dir = Path(base_dir) / f"Dataset{int(dataset_ID):03d}"
    images_dir = dataset_dir / 'imagesTr'
    labels_dir = dataset_dir / 'labelsTr'

    image_array = subject.img[tio.DATA].squeeze().numpy()
    image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255.0
    image_array = image_array.astype(np.uint8)
    image_filename = f'patient_{index:04d}_0000{file_ending}'  # Assuming single channel (MIP-PET)
    cv2.imwrite(str(images_dir / image_filename), image_array[0, :, :])

    # Label processing
    label_array = subject.label[tio.DATA].squeeze().numpy()
    binary_label_array = (label_array > 0).astype(np.uint8)  # Convert to binary (0 and 1)
    label_filename = f'patient_{index:04d}{file_ending}'
    cv2.imwrite(str(labels_dir / label_filename), binary_label_array[0, :, :])

    del image_array, label_array, binary_label_array
    gc.collect()

def save_nnUNet_raw_validation(dataset, base_dir, dataset_ID, num_subjects, file_ending='.png'):
    # Adjust dataset directory naming convention
    dataset_dir = Path(base_dir) / f"Dataset{int(dataset_ID):03d}"
    images_dir = dataset_dir / 'imagesTs'

    # Create directories
    os.makedirs(images_dir, exist_ok=True)

    # Save images and labels with correct naming convention
    print(f'Saving {num_subjects} subjects to {dataset_dir}...')
    for i, subject in tqdm(enumerate(dataset), total=len(dataset)):
        # Image processing
        image_array = subject.img[tio.DATA].squeeze().numpy()
        image_array = (image_array - image_array.min()) / (image_array.max() - image_array.min()) * 255.0
        image_array = image_array.astype(np.uint8)
        image_filename = f'patient_{i:04d}_0000{file_ending}'  # Assuming single channel (MIP-PET)
        cv2.imwrite(str(images_dir / image_filename), image_array[0, :, :])

        del image_array
        gc.collect()
    
    create_json_file(dataset_dir, num_subjects, file_ending=file_ending)


def plot_example(dataset_example):

    one_subject = dataset_example
    image_array = one_subject.img[tio.DATA].squeeze().numpy()
    label_array = one_subject.label[tio.DATA].squeeze().numpy()

    slice_idx = 0
    image_slice = image_array[slice_idx, :, :]
    label_slice = label_array[slice_idx, :, :]

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_slice, cmap='gray')
    axes[0].set_title('Image')
    axes[0].axis('off')

    axes[1].imshow(label_slice, cmap='gray')
    axes[1].set_title('Label')
    axes[1].axis('off')

    plt.show()

"""
Composition function
"""    

def torchio_compose_train(image_paths, label_paths, control_paths, 
                          invert_colors=True, 
                          cropsize = (1024,1024), 
                          train_size = False, 
                          save_inverted_imgs = False, 
                          include_controls = False,
                          dataset_ID = None):
    assert len(image_paths) == len(label_paths)

    original_size = len(image_paths)
    control_size = len(control_paths)

    print(f'Found {original_size} subjects')

    subjects = []
    for (image_path, label_path) in zip(image_paths, label_paths):
        subject = tio.Subject(
            img=tio.ScalarImage(image_path),
            label=tio.LabelMap(label_path),
        )
        subjects.append(subject)

    dataset = tio.SubjectsDataset(subjects)

    # invert colors

    if invert_colors:
        max_value = 255

        subjects_new = []

        for i, subject in enumerate(dataset):
            image_array = subject.img[tio.DATA].squeeze().numpy()
            label_array = subject.label[tio.DATA].squeeze().numpy()

            inverted_image_array = max_value - image_array

            inverted_image_tensor = torch.from_numpy(inverted_image_array).to(torch.uint8)

            subject.img[tio.DATA] = inverted_image_tensor.unsqueeze(-1)
            subject.label[tio.DATA] = torch.from_numpy(label_array).unsqueeze(-1)

            subjects_new.append(subject)

        dataset = tio.SubjectsDataset(subjects_new)

        if save_inverted_imgs:
            inverted_images_dir = 'inverted_imgs'
            save_inverted_images(dataset, inverted_images_dir)
    
    # crop and pad

    target_shape = cropsize[0], cropsize[1], 1
    crop_pad = tio.CropOrPad(target_shape)

    subjects_new = []

    for subject in dataset:
        subject = crop_pad(subject)
        subjects_new.append(subject)

    dataset = tio.SubjectsDataset(subjects_new)

    # histogram standardization

    inverted_imgs_dir = Path('inverted_imgs')
    inverted_imgs_paths = sorted(inverted_imgs_dir.glob('*.png'))

    histogram_landmarks_path = 'histogram_landmarks.npy'

    if not os.path.exists(histogram_landmarks_path):
        landmarks = tio.HistogramStandardization.train(
            images_paths=inverted_imgs_paths,
            output_path=histogram_landmarks_path
        )            
    else:
        landmarks = np.load(histogram_landmarks_path)

    landmarks_dict = {'img': landmarks}
    histogram_standardization = tio.HistogramStandardization(landmarks=landmarks_dict)
        
    subjects_new = []

    for subject in dataset:
        subject = histogram_standardization(subject)
        subjects_new.append(subject)

    dataset = tio.SubjectsDataset(subjects_new)

    # z-normalization

    znorm = tio.ZNormalization()

    subjects_new = []

    for subject in dataset:
        subject = znorm(subject)
        subjects_new.append(subject)

    dataset = tio.SubjectsDataset(subjects_new)

    save_nnUNet_raw_original(dataset, 'nnUNet_raw', dataset_ID, original_size, file_ending='.png')

    # create new subjects with augmented images

    alteration = 'small'

    if alteration == 'small':
        training_transform = tio.Compose([
            tio.RandomAnisotropy(p=0),
            tio.OneOf({
                tio.RandomAffine(scales=(0.9, 1.1),
                                degrees=(-1, 1)),
                tio.RandomElasticDeformation(max_displacement=(5, 5, 0),
                                            num_control_points=5)},
                p=1),
            tio.OneOf({
                tio.RandomBlur(std=(0,1)),
                tio.RandomNoise(std=(0.05))},
                p=0.1),
            tio.RandomBiasField(coefficients=0.25,
                                p=0)
        ])

    if train_size != False:
        if include_controls:
            augmented_size = train_size - original_size - control_size
        else:
            augmented_size = train_size - original_size
        print(f'Augmenting dataset with {augmented_size} subjects')
        j = 0
        for i in tqdm(range(augmented_size)):
            subject = dataset[j]
            augmented_subject = training_transform(subject)  # Apply transformations
            save_nnUNet_raw_augmented(augmented_subject, 'nnUNet_raw', dataset_ID, original_size+i, file_ending='.png')
            del augmented_subject  # Free memory
            gc.collect()  # Explicit garbage collection call
            j = (j + 1) % original_size  # Cycle through the dataset

    # add controls

    if include_controls:
        assert train_size >= original_size + control_size

        print(f'Adding {control_size} control subjects')

        control_subjects = []
        for control_path in control_paths:
            subject = tio.Subject(
                img=tio.ScalarImage(control_path),
                label=tio.LabelMap(control_path),
            )
            control_subjects.append(subject)

        control_dataset = tio.SubjectsDataset(control_subjects)

        # invert colors

        if invert_colors:
            max_value = 255

            subjects_new = []

            for i, subject in enumerate(control_dataset):
                image_array = subject.img[tio.DATA].squeeze().numpy()
                label_array = subject.label[tio.DATA].squeeze().numpy()

                inverted_image_array = max_value - image_array
                label_array = np.zeros_like(label_array)

                inverted_image_tensor = torch.from_numpy(inverted_image_array).to(torch.uint8)

                subject.img[tio.DATA] = inverted_image_tensor.unsqueeze(-1)
                subject.label[tio.DATA] = torch.from_numpy(label_array).unsqueeze(-1)

                subjects_new.append(subject)

            control_dataset = tio.SubjectsDataset(subjects_new)

        #check whether all labels are zero
        assert np.all(control_dataset[0].label[tio.DATA].squeeze().numpy() == 0)
        
        # crop and pad

        target_shape = cropsize[0], cropsize[1], 1

        subjects_new = []

        for subject in control_dataset:
            subject = crop_pad(subject)
            subjects_new.append(subject)

        control_dataset = tio.SubjectsDataset(subjects_new)

        # histogram standardization
            
        subjects_new = []

        for subject in control_dataset:
            subject = histogram_standardization(subject)
            subjects_new.append(subject)

        control_dataset = tio.SubjectsDataset(subjects_new)

        # z-normalization

        subjects_new = []

        for subject in control_dataset:
            subject = znorm(subject)
            subjects_new.append(subject)

        control_dataset = tio.SubjectsDataset(subjects_new)

        save_nnUNet_raw_control(control_dataset, 'nnUNet_raw', dataset_ID, control_size, augmented_size + original_size, file_ending='.png')
    
    create_json_file(Path('nnUNet_raw') / f'Dataset{int(dataset_ID):03d}', train_size, file_ending='.png')

    return True


"""
Validation function
"""

def torchio_validation_composition(image_paths, 
                          invert_colors=True,
                          cropsize = (400,991),
                          dataset_ID = None):
    print(f'Found {len(image_paths)} subjects')

    subjects = []
    for image_path in image_paths:
        subject = tio.Subject(
            img=tio.ScalarImage(image_path),
        )
        subjects.append(subject)

    dataset = tio.SubjectsDataset(subjects)

    # invert colors

    if invert_colors:
        max_value = 255

        subjects_new = []

        for i, subject in enumerate(dataset):
            image_array = subject.img[tio.DATA].squeeze().numpy()
            inverted_image_array = max_value - image_array
            inverted_image_tensor = torch.from_numpy(inverted_image_array).to(torch.uint8)
            subject.img[tio.DATA] = inverted_image_tensor.unsqueeze(-1)
            subjects_new.append(subject)

        dataset = tio.SubjectsDataset(subjects_new)
    
    # crop and pad
    
    target_shape = cropsize[0], cropsize[1], 1
    crop_pad = tio.CropOrPad(target_shape)

    subjects_new = []
    original_sizes = []

    for subject in dataset:
        original_height, original_width = subject.img[tio.DATA].shape[-3], subject.img[tio.DATA].shape[-2]
        original_sizes.append(np.array([original_height, original_width]))
        subject = crop_pad(subject)
        subjects_new.append(subject)

    dataset = tio.SubjectsDataset(subjects_new)

    # histogram standardization

    histogram_landmarks_path = 'histogram_landmarks.npy'
    landmarks = np.load(histogram_landmarks_path)
    landmarks_dict = {'img': landmarks}

    histogram_standardization = tio.HistogramStandardization(landmarks=landmarks_dict)
        
    subjects_new = []

    for subject in dataset:
        subject = histogram_standardization(subject)
        subjects_new.append(subject)

    dataset = tio.SubjectsDataset(subjects_new)

    # z-normalization

    znorm = tio.ZNormalization()

    subjects_new = []

    for subject in dataset:
        subject = znorm(subject)
        subjects_new.append(subject)

    dataset = tio.SubjectsDataset(subjects_new)

    if dataset_ID != None:
        save_nnUNet_raw_validation(dataset, 'nnUNet_raw', dataset_ID, len(dataset), file_ending='.png')

    return original_sizes

def process_and_crop_labels(label_files, original_sizes, save_dir='data_validation/patients/labels', file_ending='.png'):
    os.makedirs(save_dir, exist_ok=True)

    index = 0
    for label_file, original_size in zip(label_files, original_sizes):
        # Load the label image
        label_image = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE)

        # Crop to get the central part of the image
        current_height, current_width = label_image.shape
        original_height, original_width = original_size
        x_start = (current_width - original_width) // 2
        y_start = (current_height - original_height) // 2
        cropped_label_image = label_image[y_start:y_start + original_height, x_start:x_start + original_width]

        # Change color scale to black and white (0 and 255)
        cropped_label_image = np.where(cropped_label_image > 0, 255, 0).astype(np.uint8)

        #rotate the image
        rotated_label_image = np.rot90(cropped_label_image, 3)
        
        #mirror the image
        rotated_label_image = np.flip(rotated_label_image, 1)

        # Save the processed label
        label_filename = f'segmentation_{index:03d}{file_ending}'
        cv2.imwrite(os.path.join(save_dir, label_filename), rotated_label_image)

        index += 1

    print(f"Processed labels saved to {save_dir}")


"""
One Image
"""

def torchio_validation_single_image(image_array, base_dir,
                          invert_colors=True,
                          cropsize=(400,991),
                          dataset_ID=None):
    print(f'Processing image: {image_array}')

    # Create a single subject
    subject = tio.Subject(
        img=tio.ScalarImage(image_array),
    )

    # Create dataset with a single subject
    dataset = tio.SubjectsDataset([subject])

    # invert colors
    if invert_colors:
        max_value = 255
        image_array = subject.img[tio.DATA].squeeze().numpy()
        inverted_image_array = max_value - image_array
        inverted_image_tensor = torch.from_numpy(inverted_image_array).to(torch.uint8)
        subject.img[tio.DATA] = inverted_image_tensor.unsqueeze(-1)
        dataset = tio.SubjectsDataset([subject])

    # crop and pad
    target_shape = cropsize[0], cropsize[1], 1
    crop_pad = tio.CropOrPad(target_shape)
    original_height, original_width = subject.img[tio.DATA].shape[-3], subject.img[tio.DATA].shape[-2]
    original_size = np.array([original_height, original_width])
    subject = crop_pad(subject)
    dataset = tio.SubjectsDataset([subject])

    # histogram standardization
    histogram_landmarks_path = base_dir + '/histogram_landmarks.npy'
    landmarks = np.load(histogram_landmarks_path)
    landmarks_dict = {'img': landmarks}
    histogram_standardization = tio.HistogramStandardization(landmarks=landmarks_dict)
    subject = histogram_standardization(subject)
    dataset = tio.SubjectsDataset([subject])

    # z-normalization
    znorm = tio.ZNormalization()
    subject = znorm(subject)
    dataset = tio.SubjectsDataset([subject])

    save_nnUNet_raw_validation(dataset, 'nnUNet_raw', dataset_ID, 1, file_ending='.png')

    return original_size, dataset 


def process_and_crop_single_label(label_file, original_size, save_dir='data_validation/patients/labels', file_ending='.png'):
    os.makedirs(save_dir, exist_ok=True)

    label_image = cv2.imread(str(label_file), cv2.IMREAD_GRAYSCALE)

    current_height, current_width = label_image.shape
    original_height, original_width = original_size
    x_start = (current_width - original_width) // 2
    y_start = (current_height - original_height) // 2
    cropped_label_image = label_image[y_start:y_start + original_height, x_start:x_start + original_width]

    cropped_label_image = np.where(cropped_label_image > 0, 255, 0).astype(np.uint8)

    rotated_label_image = np.rot90(cropped_label_image, 3)
    rotated_label_image = np.flip(rotated_label_image, 1)

    label_filename = f'segmentation_000{file_ending}'
    cv2.imwrite(os.path.join(save_dir, label_filename), rotated_label_image)

    print(f"Processed label saved to {save_dir}")