from Code.datasets.phi_dataset import PhiDataset
from Code.utils.io_utils import load_project_config

cfg = load_project_config("Code/config.yaml")
ds_tr = PhiDataset(cfg, mode="train")
ds_va = PhiDataset(cfg, mode="val")
ds_te = PhiDataset(cfg, mode="test")

print(len(ds_tr), len(ds_va), len(ds_te))
x0, y0 = ds_tr[0]
print(x0.shape, y0.item())  # очікуємо (8,), y ∈ {0,1,2,3}
