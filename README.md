# QNA

pip install torch torchvision
pip install scikit-learn

preprocessng:
# 1) PCA
python -m Code.preprocess.preprocess_pca --config config.yaml

# 2) Fit скейлера (+опціональний Z-кеш)
python -m Code.preprocess.preprocess_fit_scaler --config config.yaml

# 3) Prescreen k (оновить angles.k у config.yaml)
python -m Code.preprocess.prescreen_k --config config.yaml --split auto --k-grid 1.5,2.0,2.5,3.0,3.5 --target-clip 0.02 --use-cache auto --write-config
