import torch
import torch.nn as nn
from Code.quantum_layer.quantum_layer import QuantumLayer
from Code.logger import setup_logger

class QNNModel(nn.Module):
    """
    Гібридна квантово-класична модель для класифікації.
    Вхід: PCA-зменшені класичні ознаки (batch_size × pca_dim)
    Вихід: logits (batch_size × n_classes)
    """

    def __init__(self, config, device=None, logger=None):
        """
        Args:
            config (dict): Конфігурація моделі, включаючи quantum_layer.
            device (torch.device): CPU або CUDA.
            logger: Logger для запису подій.
        """
        super().__init__()

        self.input_dim = config['input_dim']          # Початкова розмірність (наприклад, 784 для MNIST)
        self.pca_dim = config['pca_dim']              # Кількість головних компонент (== n_qubits)
        self.n_classes = config['n_classes']
        self.device = device if device else torch.device("cpu")
        self.logger = logger or setup_logger("QNNModel")

        # 1️⃣ (Фіксований PCA — дані вже зменшені. Якщо хочеш тренований, використовуй nn.Linear)
        # self.pca_layer = nn.Linear(self.input_dim, self.pca_dim)
        # self.logger.info("PCA layer: %d → %d", self.input_dim, self.pca_dim)

        # 2️⃣ Квантовий шар (QuantumLayer)
        quantum_layer_config = config['quantum_layer']
        assert self.pca_dim == quantum_layer_config['n_qubits'], \
            "pca_dim має дорівнювати n_qubits для коректного узгодження розмірності!"
        self.quantum_layer = QuantumLayer(quantum_layer_config, device=self.device, logger=self.logger)

        # 3️⃣ Класичний вихідний шар (Linear) для класифікації
        self.output_layer = nn.Linear(self.pca_dim, self.n_classes)
        self.logger.info("QNNModel initialized: %s", config)

    def forward(self, x):
        """
        Forward pass: QuantumLayer → Output Linear → logits
        Args:
            x (torch.Tensor): PCA-зменшені дані (batch_size × pca_dim)
        Returns:
            torch.Tensor: logits (batch_size × n_classes)
        """
        self.logger.debug("Forward pass QNNModel. Input shape: %s", x.shape)
        x = x.to(self.device)

        # Quantum layer
        q_out = self.quantum_layer(x)            # (batch_size × pca_dim)
        # Output Linear layer
        logits = self.output_layer(q_out)        # (batch_size × n_classes)

        return logits

    def get_quantum_params(self):
        """Повертає параметри тільки квантового шару (для QNA-Adam чи окремих експериментів)."""
        return self.quantum_layer.parameters()

    def to(self, device):
        """Явний метод для перенесення моделі на потрібний пристрій."""
        super().to(device)
        self.quantum_layer.to(device)
        self.output_layer.to(device)
        self.device = device
        return self
