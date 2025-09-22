# server.py
import flwr as fl
import tensorflow as tf
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional, Union
from flwr.common import FitRes, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_proxy import ClientProxy
from data_loader import load_and_partition_data
from models import create_simple_cnn

# --- Helper function and Custom Strategy for Weight-Printing ---

def summarize_weights(weights: List[np.ndarray]) -> str:
    """A helper function to print a summary of the model weights."""
    if weights and len(weights) > 0:
        return f"{np.sum(weights[0]):.4f}"
    return "N/A"

class CustomFedAvg(fl.server.strategy.FedAvg):
    """A custom strategy to print weights before and after aggregation."""
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        
        # Get weights from clients before aggregation
        weights_before = [parameters_to_ndarrays(fit_res.parameters) for _, fit_res in results]
        weight_summaries = [summarize_weights(w) for w in weights_before]
        print(f"\n--- Round {server_round}: Aggregation ---")
        print(f"Weights before aggregation (summaries): {weight_summaries}")

        # Call the parent class's aggregate_fit to do the actual aggregation
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)

        # Print a summary of the weights after aggregation
        if aggregated_parameters is not None:
            weights_after = parameters_to_ndarrays(aggregated_parameters)
            print(f"Weight after aggregation (summary): {summarize_weights(weights_after)}")
        
        return aggregated_parameters, aggregated_metrics

# --- Load Data and Define Server-Side Evaluation ---

print("Loading centralized test set for the server...")
_, (x_test, y_test) = load_and_partition_data('data/Brain_MRI/sampled_dataset/Training/')
print("Test set loaded.")

def get_evaluate_fn():
    """Return an evaluation function for server-side evaluation."""
    model = create_simple_cnn(input_shape=(256, 256, 3), num_classes=4)

    def evaluate(server_round: int, parameters: fl.common.NDArrays, config: Dict[str, Scalar]):
        model.set_weights(parameters)
        loss, accuracy = model.evaluate(x_test, y_test, batch_size=32, verbose=0)
        
        print(f"--- Server-side evaluation round {server_round} ---")
        print(f"Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")
        print("-------------------------------------------------")
        
        return loss, {"accuracy": accuracy}

    return evaluate

# --- Define the Strategy ---

strategy = CustomFedAvg(
    fraction_fit=1.0,
    fraction_evaluate=0.0,  # Disable client-side evaluation
    min_fit_clients=5,
    min_available_clients=5,
    evaluate_fn=get_evaluate_fn(),  # Enable server-side evaluation
)

# --- Start the Server and Save Results ---

print("Starting Flower server...")
history = fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=30),
    strategy=strategy
)
print("Flower server stopped.")
