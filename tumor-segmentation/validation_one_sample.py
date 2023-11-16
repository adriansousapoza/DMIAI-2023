import os
import time
import cv2
import numpy as np
from pathlib import Path
import subprocess
from utils import dice_score
import importlib
import utils_validation
#reload utils_validation    
importlib.reload(utils_validation)
from utils_validation import torchio_validation_single_image, process_and_crop_single_label
import torchio as tio

base_path = os.getcwd()
print(base_path)
# if os.path.basename(base_path) != 'DMIAI_2023':
#     base_path = os.path.join(base_path, 'DMIAI_2023')
#     os.chdir(base_path)
if os.path.basename(base_path) != 'tumor-segmentation':
    base_path = os.path.join(base_path, 'tumor-segmentation')
    os.chdir(base_path)
assert os.path.basename(base_path) == 'tumor-segmentation'    

os.environ['nnUNet_raw'] = os.path.join(base_path, 'nnUNet_raw')
os.environ['nnUNet_preprocessed'] = os.path.join(base_path, 'nnUNet_preprocessed')
os.environ['nnUNet_results'] = os.path.join(base_path, 'nnUNet_results')

from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.paths import nnUNet_results, nnUNet_raw
import torch
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO

# get image from DM-i-AI-2023/tumor-segmentation/incoming_images

def single_image_pipeline(image_path, base_path, dataset_ID=None):
    start_time = time.time()

    # Setting up directories and environment variables
    os.chdir(base_path)
    print(f"Current working directory: {os.getcwd()}")

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
    print("Image shape:", img.shape)
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




    
    """
    # Define and run the prediction command
    command = 'nnUNetv2_predict -i nnUNet_raw/Dataset001/imagesTs -o nnUNet_raw/Dataset001/labelsTs -d 1 -c 2d -f 5 -device cpu'
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.getcwd())
    """

    """
    # Continuously read and print output
    while True:
        output = process.stdout.readline()
        if output == b'' and process.poll() is not None:
            break
        if output:
            print(output.strip().decode())

    # Check for any errors
    stderr = process.stderr.read()
    if stderr:
        print("Error:", stderr.decode())

    process.wait()
    """
    

    # Process the predicted label
    validation_dir_name = os.path.join(base_path, 'nnUNet_raw/Dataset001/labelsTs')
    validation_dir = Path(validation_dir_name)
    validation_path = next(validation_dir.glob('*.png'))  # Assuming only one predicted label
    process_and_crop_single_label(validation_path, 
                                  original_size,
                                  save_dir=os.path.join(dataset_dir_name, 'patients/labels'))
    
    # Calculate elapsed time
    end_time = time.time()
    print("Time elapsed:", end_time - start_time)

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
    dataset_dir = Path(dataset_dir_name)
    images_dir = dataset_dir / 'patients/imgs'
    image_paths = sorted(images_dir.glob('*.png'))

    for i in range(1):
        image_path = image_paths[i]
        print(f"Processing image {i+1}: {image_path}")
        validation_path = single_image_pipeline(str(image_path), base_path, dataset_ID)
        print(f"Processed image {i+1}, output at {validation_path}")


    # load img
    img_done_filenae = 'data_validation/patients/labels/patient_0000.png'
    img = cv2.imread(img_filename)


    
