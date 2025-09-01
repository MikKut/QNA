# quantum_layer.py

import torch
import torch.nn as nn
import pennylane as qml
from Code.utils.utils import create_quantum_device, expectation_to_tensor, assert_shape
from Code.logger import setup_logger

class QuantumLayer(nn.Module):
    """
    QuantumLayer відповідає за квантове кодування PCA-даних та застосування
    hardware-efficient ansatz для обчислення expectation values.

    Вхід: batch класичних даних (batch_size × n_qubits)
    Вихід: expectation values (batch_size × n_qubits)
    """

    def __init__(self, config, device=None, logger=None):
        """
        Args:
            config (dict): Конфігурація шару (n_qubits, n_layers, shots, encoding_type, measurement).
            device (torch.device): CPU або CUDA.
            logger: Користувацький або внутрішній логер.
        """
        super().__init__()

        # Параметри з конфігурації
        self.n_qubits = config['n_qubits']
        self.n_layers = config['n_layers']
        self.shots = config['shots']
        self.topology = config['topology']
        self.encoding_type = config.get('encoding_type', 'Ry')
        self.measurement = config.get('measurement', 'PauliZ')

        # Пристрій PyTorch
        self.torch_device = device if device else torch.device('cpu')

        # Створюємо логер
        self.logger = logger or setup_logger(logger_name="QuantumLayer")
        self.logger.info("QuantumLayer initialization with config: %s", config)

        # Створюємо quantum device (через utils.py)
        self.q_device = create_quantum_device(self.n_qubits, self.shots)

        # Ініціалізація параметрів θ як torch.nn.Parameter
        self.theta = nn.Parameter(
            torch.randn(self.n_layers, self.n_qubits, device=self.torch_device),
            requires_grad=True
        )

        # Ініціалізація квантової схеми (QNode)
        self.qnode = qml.QNode(self.quantum_circuit, self.q_device, interface='torch')

    def quantum_circuit(self, inputs, theta):
        """
        Квантова схема (QNode): кодування + ansatz + вимірювання.

        Args:
            inputs (torch.Tensor): PCA-дані (n_qubits,)
            theta (torch.Tensor): параметри квантового шару (n_layers × n_qubits)

        Returns:
            List[expectation]: список expectation values з кубітів
        """
        # Перевірка розмірів
        assert_shape(inputs, (self.n_qubits,))
        assert_shape(theta, (self.n_layers, self.n_qubits))

        # Quantum embedding PCA-даних у rotation gates
        for idx, val in enumerate(inputs):
            if self.encoding_type == 'Ry':
                qml.RY(val, wires=idx)
            elif self.encoding_type == 'Rx':
                qml.RX(val, wires=idx)
            elif self.encoding_type == 'Rz':
                qml.RZ(val, wires=idx)
            else:
                raise ValueError(f"Unknown encoding type: {self.encoding_type}")

        # Hardware-efficient ansatz
        for layer in range(self.n_layers):
            # Rotation block
            for qubit in range(self.n_qubits):
                qml.RY(theta[layer, qubit], wires=qubit)

            if self.topology == 'ring':
                    for qubit in range(self.n_qubits - 1):
                        qml.CNOT(wires=[qubit, qubit + 1])
                    qml.CNOT(wires=[self.n_qubits - 1, 0])
            elif self.topology == 'linear':
                for qubit in range(self.n_qubits - 1):
                    qml.CNOT(wires=[qubit, qubit + 1])
            else:
                raise ValueError(f"Unknown topology type: {self.topology}")


        # Вимірювання expectation values (за замовчуванням PauliZ)
        expvals = []
        for qubit in range(self.n_qubits):
            if self.measurement == 'PauliZ':
                expvals.append(qml.expval(qml.PauliZ(qubit)))
            elif self.measurement == 'PauliX':
                expvals.append(qml.expval(qml.PauliX(qubit)))
            elif self.measurement == 'PauliY':
                expvals.append(qml.expval(qml.PauliY(qubit)))
            else:
                raise ValueError(f"Unknown measurement type: {self.measurement}")

        return expvals

    def forward(self, x_batch):
        """
        Forward-pass QuantumLayer для batch-обробки.

        Args:
            x_batch (torch.Tensor): batch PCA-даних (batch_size × n_qubits)

        Returns:
            torch.Tensor: expectation values (batch_size × n_qubits)
        """
        batch_size = x_batch.size(0)
        self.logger.debug("QuantumLayer forward pass with batch size: %d", batch_size)

        outputs = []
        for idx in range(batch_size):
            inputs = x_batch[idx]
            expvals = self.qnode(inputs, self.theta)
            expvals_tensor = expectation_to_tensor(expvals, device=self.torch_device)
            outputs.append(expvals_tensor)

        # Стекуємо результати в тензор (batch_size × n_qubits)
        quantum_out = torch.stack(outputs)

        # Перевіряємо розміри
        assert_shape(quantum_out, (batch_size, self.n_qubits))

        return quantum_out
