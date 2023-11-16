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

    filepath = Path('nnUNet_raw/Dataset001/imagesTs/patient_0000.png')

    return original_size, filepath 


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