import os
import time
import cv2
import numpy as np
from pathlib import Path
import importlib
import utils_validation
importlib.reload(utils_validation)
from utils_validation import torchio_validation_single_image, process_and_crop_single_label
from utils import validate_segmentation

base_path = os.getcwd()
if os.path.basename(base_path) != 'workspace':
    if os.path.basename(base_path) != 'DMIAI_2023':
        base_path = os.path.join(base_path, 'DMIAI_2023')
        os.chdir(base_path)
    if os.path.basename(base_path) != 'tumor-segmentation':
        base_path = os.path.join(base_path, 'tumor-segmentation')
        os.chdir(base_path)
    assert os.path.basename(base_path) == 'tumor-segmentation'   

print(f"Current working directory: {os.getcwd()}") 

os.environ['nnUNet_raw'] = os.path.join(base_path, 'nnUNet_raw')
os.environ['nnUNet_preprocessed'] = os.path.join(base_path, 'nnUNet_preprocessed')
os.environ['nnUNet_results'] = os.path.join(base_path, 'nnUNet_results')

from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

predictor = nnUNetPredictor(
    tile_step_size=0.5,
    use_gaussian=True,
    use_mirroring=True,
    perform_everything_on_gpu=False,
    device=torch.device('cpu', 0),
    verbose=False,
    verbose_preprocessing=False,
    allow_tqdm=True
)

dataset_ID = 2
name_dataset = f'Dataset{dataset_ID:03d}'

predictor.initialize_from_trained_model_folder(
    join(nnUNet_results, f'{name_dataset}/nnUNetTrainer__nnUNetPlans__2d'),
    use_folds=(5,),
    checkpoint_name='checkpoint_best.pth',
)

def single_image_pipeline(image_array, base_path, predictor, index, dataset_ID=dataset_ID):

    start_time = time.time()
    formatted_index = f'{index:03d}'
    cv2.imwrite(os.path.join(base_path, f'data_validation/patients/imgs/patient_{formatted_index}.png'), image_array)
    
    image_path = os.path.join(base_path, f'data_validation/patients/imgs/patient_{formatted_index}.png')

    dataset_dir_name = os.path.join(base_path, 'data_validation')
    dataset_dir = Path(dataset_dir_name)
    images_dir = dataset_dir / 'patients/imgs'

    # Validation of the image
    original_size, filepath = torchio_validation_single_image(image_path, 
                                                             base_dir=base_path,
                                                             invert_colors=True,
                                                             cropsize=(400,991),
                                                             dataset_ID=dataset_ID)

    # Preparing for nnUNet prediction
    img, props = SimpleITKIO().read_images([filepath])
    predicted_mask = predictor.predict_single_npy_array(img, props, None, None, True)

    mask = predicted_mask[1]
    mask_channel = mask[1]
    mask_channel_2d = mask_channel.squeeze()

    # Convert the mask to uint8
    mask_uint8 = (mask_channel_2d * 255).astype(np.uint8)

    # Process the predicted label
    validation_image = process_and_crop_single_label(mask_uint8, original_size)

    end_time = time.time()
    print("Time elapsed:", end_time - start_time)

    return validation_image

if __name__ == '__main__':
    dataset_dir_name = os.path.join(base_path, 'data')
    dataset_dir = Path(dataset_dir_name)
    images_dir = dataset_dir / 'patients/imgs'
    image_paths = sorted(images_dir.glob('*.png'))

    for index in range(10):
        image_path = image_paths[index]
        image_array = cv2.imread(str(image_path))
        print("Image array shape:", image_array.shape)
        print(f"Processing image {index+1}: {image_path}")
        final_segmentation = single_image_pipeline(image_array, base_path, predictor, index, dataset_ID)

        validate_segmentation(image_array, final_segmentation)
        
