import cv2
import numpy as np

def preprocess_image(image_path, blur=True, normalize=True, clip=True, equalize_hist=True, resize=True,
                     target_size=(400, 400), clip_limit=0.03, grid_size=(8, 8)):
    # Read the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Normalize the image to range [0, 255] if normalization is true
    if normalize:
        image = cv2.normalize(image, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    # Noise reduction with Gaussian blur if blur is true
    if blur:
        image = cv2.GaussianBlur(image, (3, 3), 0)

    # Clip the intensity and apply Contrast Limited Adaptive Histogram Equalization (CLAHE) if clip is true
    if clip and equalize_hist:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        image = clahe.apply(image)
    elif equalize_hist:  # Apply Histogram Equalization without clipping
        image = cv2.equalizeHist(image)

    # Resize the image to the target size if resize is true
    if resize:
        image = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    
    return image


# Example usage for a single image:
# processed_image = preprocess_image('path/to/image.png', blur=True, normalize=True, clip=True, equalize_hist=True, resize=True)
