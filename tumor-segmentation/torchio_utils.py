import os
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchio as tio
from pathlib import Path
import shutil
import json



# Content of your dataset.json
dataset_json_content = {
    "channel_names": {
        "0": "MIP-PET"
    },
    "labels": {
        "background": 0,
        "TargetRegion": 1
    },
    "numTraining": 0,  # This will be updated later
    "file_ending": ".png"
}


def save_inverted_images(dataset, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for i, subject in enumerate(dataset):
        image_array = subject.img[tio.DATA].squeeze().numpy()
        slice_index = 0
        image_slice = image_array[slice_index, :, :]
        image_pil = Image.fromarray(image_slice)
        image_pil.save(os.path.join(save_dir, f'patient_{i}.png'))

def save_nnUNet_raw(dataset, save_dir, num_subjects, dataset_json_content):
    os.makedirs(save_dir, exist_ok=True)
    images_dir = os.path.join(save_dir, 'imagesTr')
    labels_dir = os.path.join(save_dir, 'labelsTr')
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)

    for i, subject in enumerate(dataset):
        # Convert and normalize image data
        image_array = subject.img[tio.DATA].squeeze().numpy()
        image_array -= image_array.min()  # Normalize to 0
        image_array /= image_array.max()  # Normalize to 1
        image_array *= 255.0  # Scale to 0-255
        image_array = image_array.astype(np.uint8)  # Convert to uint8
        slice_index = 0
        image_slice = image_array[slice_index, :, :]
        image_pil = Image.fromarray(image_slice)

        # Process label data
        label_array = subject.label[tio.DATA].squeeze().numpy()
        label_slice = label_array[slice_index, :, :]
        label_pil = Image.fromarray(label_slice.astype(np.uint8))

        # Save the images and labels
        image_pil.save(os.path.join(images_dir, f'patient_{i}.png'))
        label_pil.save(os.path.join(labels_dir, f'patient_{i}.png'))

    dataset_json_path = os.path.join(save_dir, 'dataset.json')
    with open(dataset_json_path, 'w') as f:
        json.dump(dataset_json_content, f, indent=4)

    # Update the number of training images in dataset.json
    with open(dataset_json_path, 'r') as f:
        data = json.load(f)
        data['numTraining'] = num_subjects
    with open(dataset_json_path, 'w') as f:
        json.dump(data, f, indent=4)

    # Create a zip file containing the nnUNet_raw data including dataset.json
    shutil.make_archive(save_dir, 'zip', save_dir)


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
                        save_training_dataset = False):
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

    if train_size != False:
        subjects_augmented = []
        j = 0

        for i in range(train_size-len(dataset)):
            subject = dataset[j]
            subject = training_transform(subject)
            subjects_augmented.append(subject)
            j += 1
            if j == len(dataset):
                j = 0
    
        all_subjects = subjects_original + subjects_augmented
        dataset = tio.SubjectsDataset(all_subjects)

    if save_training_dataset:
        # rescale images to 0-255 for png conversion
        rescale_intensity = tio.RescaleIntensity((0, 1))

        subjects_new = []

        for subject in dataset:
            subject = rescale_intensity(subject)
            subjects_new.append(subject)

        dataset = tio.SubjectsDataset(subjects_new)
        save_nnUNet_raw(dataset, 'nnUNet_raw/dataset001', len(dataset), dataset_json_content)
    
    return dataset


