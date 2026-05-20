from __future__ import annotations

from src.data.common_dataset import VLADataset


class SyntheticVLADataset(VLADataset):
    def __init__(self, hdf5_path="data/synthetic_demos.hdf5", horizon: int = 16):
        super().__init__(hdf5_path=hdf5_path, horizon=horizon, dataset_name="synthetic")

