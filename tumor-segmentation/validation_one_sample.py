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
print(base_path)
# if os.path.basename(base_path) != 'DMIAI_2023':
#     base_path = os.path.join(base_path, 'DMIAI_2023')
#     os.chdir(base_path)
if os.path.basename(base_path) != 'tumor-segmentation':
    base_path = os.path.join(base_path, 'tumor-segmentation')
    os.chdir(base_path)
assert os.path.basename(base_path) == 'tumor-segmentation'    
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

# get image from DM-i-AI-2023/tumor-segmentation/incoming_images

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
# initializes the network architecture, loads the checkpoint
predictor.initialize_from_trained_model_folder(
    join(nnUNet_results, 'Dataset001/nnUNetTrainer__nnUNetPlans__2d'),
    use_folds=(5,),
    checkpoint_name='checkpoint_final.pth',
)

dataset_ID = 1

def single_image_pipeline(image_array, base_path, predictor, index, dataset_ID=1):
    formatted_index = f'{index:03d}'
    cv2.imwrite(os.path.join(base_path, f'data_validation/patients/imgs/patient_{formatted_index}.png'), image_array)
    
    image_path = os.path.join(base_path, f'data_validation/patients/imgs/patient_{formatted_index}.png')
    #print shape of array
    print(cv2.imread(image_path).shape)

    start_time = time.time()

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
    img, props = SimpleITKIO().read_images([join(nnUNet_raw, 'Dataset001/imagesTs/patient_0000_0000.png')])
    predicted_mask = predictor.predict_single_npy_array(img, props, None, None, True)

    mask = predicted_mask[1]
    mask_channel = mask[1]
   # Save the predicted mask in the 'labelsTs' folder
    labels_ts_dir = os.path.join(base_path, 'nnUNet_raw/Dataset001/labelsTs')
    os.makedirs(labels_ts_dir, exist_ok=True)
    mask_filename = os.path.join(labels_ts_dir, 'patient_0000.png')

    mask_channel_2d = mask_channel.squeeze()

    # Convert the mask to uint8 and save
    mask_uint8 = (mask_channel_2d * 255).astype(np.uint8)
    cv2.imwrite(mask_filename, mask_uint8)

    # Process the predicted label
    validation_dir_name = os.path.join(base_path, 'nnUNet_raw/Dataset001/labelsTs')
    validation_dir = Path(validation_dir_name)
    validation_path = next(validation_dir.glob('*.png'))  # Assuming only one predicted label
    save_dir=os.path.join(dataset_dir_name, 'patients/labels')
    process_and_crop_single_label(validation_path, 
                                  original_size,
                                  save_dir=save_dir)
    
    # Calculate elapsed time
    end_time = time.time()
    print("Time elapsed:", end_time - start_time)

    print("Validation path:", validation_path)
    print('shape of validation image:', cv2.imread(validation_path).shape)

    validate_segmentation(image_array, cv2.imread(validation_path))

    return validation_path

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
# initializes the network architecture, loads the checkpoint
predictor.initialize_from_trained_model_folder(
    join(nnUNet_results, 'Dataset001/nnUNetTrainer__nnUNetPlans__2d'),
    use_folds=(5,),
    checkpoint_name='checkpoint_final.pth',
    )

def predict(img):
    # save img
    img_filename = 'data_validation/patients/imgs/patient_0000.png'
    cv2.imwrite(img_filename, img)
    
    # Set base path and dataset ID
    # base_path = '/home/asp/Downloads/DMIAI/DMIAI_2023/tumor-segmentation'
    dataset_ID = 1  # Assuming this is the dataset ID

    # Set up image and validation directories
    dataset_dir_name = os.path.join(base_path, 'data_validation')
if __name__ == '__main__':
    dataset_dir_name = os.path.join(base_path, 'data')
    dataset_dir = Path(dataset_dir_name)
    images_dir = dataset_dir / 'patients/imgs'
    image_paths = sorted(images_dir.glob('*.png'))

    for i in range(1):
        image_path = image_paths[i]
        print(f"Processing image {i+1}: {image_path}")
        validation_path = single_image_pipeline(str(image_path), base_path, dataset_ID)
        print(f"Processed image {i+1}, output at {validation_path}")


    # load img
    img_done_filename = 'data_validation/patients/labels/patient_0000.png'
    mask = cv2.imread(img_done_filename)

    return mask   
