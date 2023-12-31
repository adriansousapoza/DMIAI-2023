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
import torchio_utils
import time
importlib.reload(torchio_utils)
from torchio_utils import torchio_validation_composition, process_and_crop_labels

#create timer

start_time = time.time()

base_path = '/home/asp/Downloads/DMIAI/DMIAI_2023/tumor-segmentation'
os.chdir(base_path)


dataset_dir_name = base_path + '/data_validation'
dataset_dir = Path(dataset_dir_name)
images_dir = dataset_dir / 'patients/imgs'
image_paths = sorted(images_dir.glob('*.png'))

ORIGINAL_SIZE = torchio_validation_composition(image_paths, 
                                               base_dir=base_path,
                                               invert_colors=True,
                                               cropsize = (400,991),
                                               dataset_ID = 1)


os.environ['nnUNet_raw'] = os.path.join(base_path, 'nnUNet_raw')
os.environ['nnUNet_preprocessed'] = os.path.join(base_path, 'nnUNet_preprocessed')
os.environ['nnUNet_results'] = os.path.join(base_path, 'nnUNet_results')

start_time = time.time()

command = 'nnUNetv2_predict -i nnUNet_raw/Dataset001/imagesTs -o nnUNet_raw/Dataset001/labelsTs -d 1 -c 2d -f 5 -device cpu'

process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=os.getcwd())

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

# Calculate elapsed time
end_time = time.time()
print("Time elapsed:", end_time - start_time)

validation_dir_name = base_path + '/nnUNet_raw/Dataset001/labelsTs'
validation_dir = Path(validation_dir_name)
validation_paths = sorted(validation_dir.glob('*.png'))

print('Number of validation images: ', len(validation_paths))

process_and_crop_labels(validation_paths,
                        ORIGINAL_SIZE,
                        save_dir='data_validation/patients/labels')



dice_scores = []
for i in range(0, 10):
    index_patient = '00' + str(i)
    seg_pred_path = f'data_validation/patients/labels/segmentation_{index_patient}.png'
    seg_true_path = f'data/patients/labels/segmentation_{index_patient}.png'

    seg_pred = cv2.imread(seg_pred_path)
    seg_true = cv2.imread(seg_true_path)

    dice_scores.append(dice_score(seg_true, seg_pred))

print('Average Dice Score: ', np.mean(dice_scores))