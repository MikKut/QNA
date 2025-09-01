import torch
from Code.quantum_layer.qnn_model import QNNModel
from Code.utils.utils import load_config
import logger
# Завантаж дані
config = load_config()
X = torch.load(r'D:\Code\QNA\QNA\data\\' + config["features_train_name"] + '.pt')
model = QNNModel(config)
output = model(X[:32])
print(output.shape)
