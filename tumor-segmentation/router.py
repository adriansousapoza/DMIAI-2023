import numpy as np
from loguru import logger
from fastapi import APIRouter
from models.dtos import PredictRequestDto, PredictResponseDto
from utils import validate_segmentation, encode_request, decode_request

# our
import os
import torch
from torch.utils.data import random_split
import torchio as tio
from pathlib import Path
import importlib
from utils import validate_segmentation, plot_prediction, dice_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cv2
import numpy as np
import subprocess
#import torchio_utils
import time
#importlib.reload(torchio_utils)
#from torchio_utils import torchio_validation_composition, process_and_crop_labels

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
    join(nnUNet_results, 'Dataset003/nnUNetTrainer__nnUNetPlans__2d'),
    use_folds=(5,),
    checkpoint_name='checkpoint_best.pth',
    )
dataset_ID = 3  # Assuming this is the dataset ID

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

counter = 0
def predict(img):

    validation_image = single_image_pipeline(img, base_path, dataset_ID)

    mask = validation_image
    assert mask is not None
    assert type(mask) == type(np.array([0]))
    return mask

router = APIRouter()

@router.post('/predict', response_model=PredictResponseDto)
def predict_endpoint(request: PredictRequestDto):
    # Decode request str to numpy array
    img: np.ndarray = decode_request(request)

    # squeeze image
    # img = cv2.imread('data/patients/imgs/patient_000.png')

    predicted_segmentation = predict(img)

    # Validate segmentation format
    validate_segmentation(img, predicted_segmentation)

    # Encode the segmentation array to a str
    encoded_segmentation = encode_request(predicted_segmentation)

    # Return the encoded segmentation to the validation/evalution service
    response = PredictResponseDto(
        img=encoded_segmentation
    )
    return response