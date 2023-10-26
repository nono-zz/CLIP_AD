import numpy as np
from skimage.segmentation import slic, mark_boundaries
from PIL import Image
import random
from skimage.metrics import mean_squared_error


def superpixel_segmentation(image_path, num_superpixels=100, mask_value=None):
    # Load the image
    image = np.array(Image.open(image_path))

    # Apply superpixel segmentation using SLIC
    segments = slic(image, n_segments=num_superpixels, compactness=10)

    # Retrieve the desired superpixel region
    if mask_value == None:
        mask_value = random.randint(segments.min(), segments.max())
    superpixel_mask = (segments == mask_value).astype(np.uint8)
    

    return image, superpixel_mask

def pad_and_resize(image, mask, target_size=(512, 512)):
    # Convert to PIL Image
    image_pil = Image.fromarray(image)
    mask_pil = Image.fromarray(mask * 255)  # Convert binary mask to 0-255 range

    # Pad to resize to an integer multiple of 32
    # padded_image = np.array(pad_to_multiple_of_32(image_pil))
    # padded_mask = np.array(pad_to_multiple_of_32(mask_pil))
    padded_image = np.array(image_pil)
    padded_mask = np.array(mask_pil)
    
    # Resize to the target size
    resized_image = Image.fromarray(padded_image).resize(target_size)
    resized_mask = Image.fromarray(padded_mask).resize(target_size)

    return resized_image, resized_mask

def pad_to_multiple_of_32(image):
    width, height = image.size
    new_width = int(np.ceil(width / 32) * 32)
    new_height = int(np.ceil(height / 32) * 32)

    padded_image = Image.new("RGB", (new_width, new_height), (0, 0, 0))
    padded_image.paste(image, ((new_width - width) // 2, (new_height - height) // 2))

    return padded_image

def image_diference_check(init_image_array, result_image_array):
    mse_value = mean_squared_error(init_image_array, result_image_array)
    print(mse_value)
    if mse_value > 50:
        return True
    else:
        False
    # return mse_value
    

# Example usage:
CATEGORY = 'bottle'
image_path = f"/home/zhaoxiang/dataset/mvtec_anomaly_detection/{CATEGORY}/train/good/024.png"
num_superpixels = 100
target_superpixel = None
target_size = (512, 512)

image, superpixel_mask = superpixel_segmentation(image_path, num_superpixels, target_superpixel)
resized_image, resized_mask = pad_and_resize(image, superpixel_mask, target_size)

resized_image.save('/sda/zhaoxiang_sda/outputs/inpainting/resized_image.jpg')
resized_mask.save('/sda/zhaoxiang_sda/outputs/inpainting/resized_mask.jpg')
print('done')