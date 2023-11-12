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


# Content of your dataset.json
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
        image_array = subject.img[tio.DATA].squeeze().numpy()
        inverted_image_array = 255 - image_array
        inverted_image_array = inverted_image_array.astype(np.uint8)
        inverted_image_pil = Image.fromarray(inverted_image_array[0, :, :])
        inverted_image_filename = f'patient_{i:04d}.png'
        inverted_image_pil.save(inverted_images_dir / inverted_image_filename)


def save_nnUNet_raw(dataset, base_dir, dataset_ID, num_subjects, file_ending='.png'):
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
        image_pil = Image.fromarray(image_array[0, :, :])
        image_filename = f'patient_{i:04d}_0000{file_ending}'  # Assuming single channel (MIP-PET)
        image_pil.save(images_dir / image_filename)

        # Label processing
        label_array = subject.label[tio.DATA].squeeze().numpy()
        label_pil = Image.fromarray(label_array[0, :, :].astype(np.uint8))
        label_filename = f'patient_{i:04d}{file_ending}'
        label_pil.save(labels_dir / label_filename)

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

    # Create a zip file containing the nnUNet_raw data including dataset.json
    #shutil.make_archive(save_dir, 'zip', save_dir)


def plot_example(image_slice, label_slice):
    # Plotting the images
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(image_slice, cmap='gray')
    axes[0].set_title('Image')
    axes[0].axis('off')

    axes[1].imshow(label_slice, cmap='gray')
    axes[1].set_title('Label')
    axes[1].axis('off')

    plt.show()

def torchio_compose_train(image_paths, label_paths, control_paths, 
                        invert_colors=True, cropsize = (1024,1024), train_size = False, create_inverted_images = False, hist_standardization = True,
                        dataset_ID = None):
    assert len(image_paths) == len(label_paths)

    print(f'Found {len(image_paths)} subjects')

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

        if create_inverted_images:
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

    if hist_standardization == False:
        # rescale intensity

        rescale_intensity = tio.RescaleIntensity((0, 1), percentiles=(0.5, 99.5))

        subjects_new = []

        for subject in dataset:
            subject = rescale_intensity(subject)
            subjects_new.append(subject)

        dataset = tio.SubjectsDataset(subjects_new)

    else:
        # histogram standardization

        inverted_imgs_dir = Path('inverted_imgs')
        inverted_imgs_paths = sorted(inverted_imgs_dir.glob('*.png'))

        histogram_landmarks_path = 'histogram_landmarks.npy'

        landmarks = tio.HistogramStandardization.train(
            images_paths=inverted_imgs_paths,
            output_path=histogram_landmarks_path
        )
        landmarks = np.load(histogram_landmarks_path)
        print('Landmarks:', landmarks)
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

    # create new subjects with augmented images

    subjects_original = subjects_new

    training_transform = tio.Compose([
        tio.RandomAnisotropy(p=0.25),
        tio.OneOf({
            tio.RandomAffine(scales=(1, 1.5),
                            degrees=(-10, 10)),
            tio.RandomElasticDeformation(max_displacement=(10, 10, 0),
                                        num_control_points=10)},
            p=1),
        tio.OneOf({
            tio.RandomBlur(std=(0,2)),
            tio.RandomNoise(std=(0,0.2))},
            p=0.25),
        tio.RandomBiasField(coefficients=0.25,
                            p=0.1)
    ])

    print(f'Augmenting {train_size-len(dataset)} subjects')
    if train_size != False:
        subjects_augmented = []
        j = 0

        for i in tqdm(range(train_size-len(dataset))):
            subject = dataset[j]
            subject = training_transform(subject)
            subjects_augmented.append(subject)
            j += 1
            if j == len(dataset):
                j = 0
    
        all_subjects = subjects_original + subjects_augmented
        dataset = tio.SubjectsDataset(all_subjects)

    if dataset_ID != None:
        # rescale images to 0-255 for png conversion
        rescale_intensity = tio.RescaleIntensity((0, 1))

        subjects_new = []

        for subject in dataset:
            subject = rescale_intensity(subject)
            subjects_new.append(subject)

        dataset = tio.SubjectsDataset(subjects_new)
        channel_names = {"0": f"{dataset_ID}"}
        labels = {"background": 0, "TargetRegion": 1}
        save_nnUNet_raw(dataset, 'nnUNet_raw', dataset_ID, len(dataset), file_ending=".png")

    
    return dataset


