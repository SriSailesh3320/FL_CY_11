# data_loader.py
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

def load_and_partition_data(dataset_path, num_clients=5, img_size=(256, 256)):
    """Loads the pre-cleaned image data and partitions it among clients."""
    
    ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        labels='inferred',
        label_mode='int',
        image_size=img_size,
        batch_size=32,
        shuffle=True,
        seed=42
    )

    # Convert the dataset to NumPy arrays for partitioning
    images, labels = [], []
    for image_batch, label_batch in ds:
        images.append(image_batch.numpy())
        labels.append(label_batch.numpy())
    
    images = np.concatenate(images)
    labels = np.concatenate(labels)

    # Normalize images
    images = images.astype('float32') / 255.0

    # Split all data into a training set and a final test set
    x_train, x_test, y_train, y_test = train_test_split(images, labels, test_size=0.1, random_state=42)
    
    # Partition only the training data among the clients
    x_train_parts = np.array_split(x_train, num_clients)
    y_train_parts = np.array_split(y_train, num_clients)

    client_data = list(zip(x_train_parts, y_train_parts))
    
    return client_data, (x_test, y_test)