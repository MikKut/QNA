# test_quantum_layer_min.py
import torch
import torch.nn as nn
from Code.quantum_layer.quantum_layer import QuantumLayer
from Code.logger import setup_logger

def main():
    logger = setup_logger(level=10)  # DEBUG
    config = {
        "input_dim": 784,
        "pca_dim": 8,
        "n_classes": 10,
        "quantum_layer": {
            "n_qubits": 8,
            "n_layers": 2,
            "shots": 100,
            "encoding_type": "Ry",
            "measurement": "PauliZ",
            "topology": "ring",
            "input_clip": True,
            "clip_range": [-3.1416, 3.1416],
        }
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    layer = QuantumLayer(config, device=device, logger=logger).to(device)
    head = nn.Linear(config["quantum_layer"]["n_qubits"], 10).to(device)

    batch, n_qubits = 4, config["quantum_layer"]["n_qubits"]
    x = torch.randn(batch, n_qubits, device=device, dtype=torch.float32)
    y = torch.randint(0, 10, (batch,), device=device)

    # 1) Forward
    q_out = layer(x)
    assert q_out.shape == (batch, n_qubits)
    assert torch.isfinite(q_out).all(), "Found NaN/Inf in quantum output"
    # (значення очікувано в [-1, 1], але shots дає дисперсію — жорстко не перевіряємо)

    # 2) Backward через просту голову + CE loss
    logits = head(q_out)
    loss = nn.CrossEntropyLoss()(logits, y)
    loss.backward()

    # 3) Градієнти по theta
    theta_grad = layer.theta.grad
    assert theta_grad is not None, "theta.grad is None"
    nonzero = (theta_grad.abs() > 0).sum().item()
    print(f"Loss = {loss.item():.4f}, nonzero theta.grad = {nonzero}/{theta_grad.numel()}")

if __name__ == "__main__":
    main()
