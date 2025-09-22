# client.py
import flwr as fl
import tensorflow as tf
import numpy as np
from sklearn.metrics import precision_score, f1_score
from models import create_simple_cnn, create_vgg16_model
from data_loader import load_and_partition_data
import sys

# --- Configuration ---
NUM_CLIENTS = 5
CLIENT_ID = int(sys.argv[1])

# --- 1. Load the preprocessed dataset ---
(client_data, test_data) = load_and_partition_data('data/Brain_MRI/sampled_dataset/Training/', num_clients=NUM_CLIENTS)
(x_train, y_train) = client_data[CLIENT_ID]


# Define the Flower client class
class MedicalImagingClient(fl.client.NumPyClient):
    def __init__(self):
        self.model = create_simple_cnn(input_shape=(256, 256, 3), num_classes=4)

    def get_parameters(self, config):
        """Return the current local model parameters."""
        return self.model.get_weights()

    def fit(self, parameters, config):
        """Train the model on the local data."""
        self.model.set_weights(parameters)
        self.model.fit(x_train, y_train, epochs=1, batch_size=32, verbose=0)
        return self.model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        """Evaluate the model on the local test set."""
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(test_data[0], test_data[1], batch_size=16, verbose=0)
        
        y_pred = np.argmax(self.model.predict(test_data[0], batch_size=16), axis=1)
        precision = precision_score(test_data[1], y_pred, average='weighted', zero_division=0)
        f1 = f1_score(test_data[1], y_pred, average='weighted', zero_division=0)
        
        return loss, len(test_data[0]), {"accuracy": accuracy, "precision": precision, "f1_score": f1}

# --- 3. Start the client using the modern Flower API ---
print(f"Starting client {CLIENT_ID}...")

# Create an instance of the client and convert it to a standard client
client = MedicalImagingClient().to_client()

# Start the client
fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=client
)