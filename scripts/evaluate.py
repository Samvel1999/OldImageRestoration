import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.losses import MeanSquaredError


def restore_image(model, image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (256, 256)) / 255.0
    restored = model.predict(image[np.newaxis, ...])[0]
    return (restored * 255).astype("uint8")


if __name__ == "__main__":
    model = keras.models.load_model("checkpoints/photo_restoration_model.h5", custom_objects={'mse': MeanSquaredError()})
    test_image_path = "data/damaged/test.png"  # Update with your test image
    restored_image = restore_image(model, test_image_path)
    cv2.imwrite("output/restored_image3.png", restored_image)
