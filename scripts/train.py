import numpy as np
import os
from tensorflow.keras.optimizers import Adam
from tensorflow import keras
from unet_model import build_unet
from tensorflow.keras.losses import MeanSquaredError


class DataGenerator(keras.utils.Sequence):
    def __init__(self, clean_dir, damaged_dir, batch_size):
        self.clean_files = [os.path.join(clean_dir, f) for f in os.listdir(clean_dir) if f.endswith('.npy')]
        self.damaged_files = [os.path.join(damaged_dir, f) for f in os.listdir(damaged_dir) if f.endswith('.npy')]
        self.batch_size = batch_size

    def __len__(self):
        return len(self.clean_files) // self.batch_size

    def __getitem__(self, idx):
        batch_clean = self.clean_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_damaged = self.damaged_files[idx * self.batch_size:(idx + 1) * self.batch_size]

        clean_images = np.array([np.load(f) for f in batch_clean])
        damaged_images = np.array([np.load(f) for f in batch_damaged])

        return damaged_images, clean_images


if __name__ == "__main__":
    clean_dir = "data/preprocessed_clean"
    damaged_dir = "data/preprocessed_damaged"

    batch_size = 16
    train_gen = DataGenerator(clean_dir, damaged_dir, batch_size)

    model = build_unet()
    model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError(), metrics=['accuracy'])

    model.fit(train_gen, epochs=50)

    model.save("checkpoints/photo_restoration_model.h5")