from Code.utils.io_utils import load_yaml, load_npy
from Code.preprocess.transforms import PCScaler, AngleEncoder
import numpy as np

cfg = load_yaml('project.yaml')
stats = load_yaml(cfg['paths']['preprocess_stats'])
X = load_npy(cfg['paths']['X_train_pca'])[:32]  # 32 приклади

scaler = PCScaler.from_dict(stats)
z = scaler.transform(X)                     # (N, p)
enc = AngleEncoder(k=cfg['angles']['k'], angle_max=cfg['angles']['angle_max'])
phi = enc.encode(z)

print("z shape:", z.shape, "phi shape:", phi.shape)
print("phi range: [", float(np.min(phi)), ",", float(np.max(phi)), "]")
print("z mean/std (перші 3 осі):",
      np.round(z.mean(0)[:3],4).tolist(),
      np.round(z.std(0)[:3],4).tolist())