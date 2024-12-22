import cv2
import numpy as np
import os

def preprocess_images(input_dir, output_dir, size=(256, 256)):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)
            if image is not None:
                image = cv2.resize(image, size) / 255.0  # Normalize to [0, 1]
                image = image.astype('float32')  # Convert to float32
                np.save(os.path.join(output_dir, filename.split(".")[0] + ".npy"), image)

if __name__ == "__main__":
    preprocess_images("data/clean/img_align_celeba/img_align_celeba", "data/preprocessed_clean")
    preprocess_images("data/damaged", "data/preprocessed_damaged")