'''
utility module with helper functions and classes 

'''

# utils.py

import pennylane as qml
import torch
import yaml

def create_quantum_device(n_qubits, shots):
    return qml.device('default.qubit', wires=n_qubits, shots=shots)

def expectation_to_tensor(exp_values, device):
    return torch.tensor(exp_values, device=device, dtype=torch.float32)

def assert_shape(tensor, expected_shape):
    assert tensor.shape == expected_shape, f"Expected shape {expected_shape}, but got {tensor.shape}"

def load_config(config_path='config.yaml'):
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"Не вдалося завантажити {config_path}: {e}")
        raise
