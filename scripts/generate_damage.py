import cv2
import numpy as np
import os


def add_damage(image):
    """Simulate damage on an image (scratches, noise, etc.)"""
    height, width, _ = image.shape
    # Add scratches
    for _ in range(10):
        x1, y1 = np.random.randint(0, width), np.random.randint(0, height)
        x2, y2 = np.random.randint(0, width), np.random.randint(0, height)
        cv2.line(image, (x1, y1), (x2, y2), (255, 255, 255), thickness=1)
    # Add noise
    noise = np.random.randint(0, 50, (height, width, 3), dtype='uint8')
    image = cv2.add(image, noise)
    return image


def generate_damaged_images(clean_dir, damaged_dir):
    if not os.path.exists(damaged_dir):
        os.makedirs(damaged_dir)

    for filename in os.listdir(clean_dir):
        if filename.endswith(".jpg"):
            image_path = os.path.join(clean_dir, filename)
            image = cv2.imread(image_path)
            if image is not None:
                damaged_image = add_damage(image)
                cv2.imwrite(os.path.join(damaged_dir, filename), damaged_image)


if __name__ == "__main__":
    clean_dir = "data/clean/img_align_celeba/img_align_celeba"
    damaged_dir = "data/damaged"
    generate_damaged_images(clean_dir, damaged_dir)
