import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import Sequence


class MultiChannelDataGenerator(Sequence):
    def __init__(self, dataset_dir, batch_size=32, img_size=(256, 256)):
        self.dataset_dir = dataset_dir
        self.batch_size = batch_size
        self.img_size = img_size

        # Assuming the structure /dataset/{channel}/{image}.png
        self.channels = ['rgb8', 'infrared']
        self.filenames = os.listdir(os.path.join(dataset_dir, 'rgb8'))  # Assuming all channels have the same filenames

        # Load labels here if you have them
        self.labels = self.load_labels()

    def load_labels(self):
        # Implement this method based on how your labels are stored
        # Return a list or array where labels[i] corresponds to the label of self.filenames[i]
        return []

    def __len__(self):
        return len(self.filenames) // self.batch_size

    def __getitem__(self, idx):
        batch_filenames = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_images = []
        batch_labels = []

        for filename in batch_filenames:
            images = []
            for channel in self.channels:
                img_path = os.path.join(self.dataset_dir, channel, filename)
                img = load_img(img_path, color_mode='grayscale', target_size=self.img_size)
                img = img_to_array(img)
                images.append(img)

            # Stack the single-channel images along the last dimension to form a multi-channel image
            multi_channel_image = np.concatenate(images, axis=-1)
            batch_images.append(multi_channel_image)

            # Assuming you have implemented load_labels
            batch_labels.append(self.labels[idx * self.batch_size + len(batch_images) - 1])

        return np.array(batch_images), np.array(batch_labels)

    def on_epoch_end(self):
        # Shuffle the data if necessary
        pass


# Usage
dataset_dir = 'C:\\Users\\walra\\Stichting Hogeschool Utrecht\\Data Sience MNLE - General\\pools\\Beeldmateriaal_ortho_2023'
batch_size = 32
img_size = (256, 256)  # Adjust based on your dataset

data_generator = MultiChannelDataGenerator(dataset_dir, batch_size=batch_size, img_size=img_size)
print(len(data_generator))
# Now you can use this data_generator with Keras model's fit method
