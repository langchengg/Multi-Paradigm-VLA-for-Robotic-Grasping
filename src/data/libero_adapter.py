from __future__ import annotations

import warnings
from pathlib import Path

from src.data.common_dataset import VLADataset


def libero_available() -> bool:
    try:
        import libero  # noqa: F401
        return True
    except Exception:
        try:
            import libero.libero  # noqa: F401
            return True
        except Exception:
            return False


class LiberoVLADataset(VLADataset):
    """Optional LIBERO adapter.

    If a converted LIBERO HDF5 file is supplied, it is exposed through the same
    common dataset interface. Native LIBERO imports are kept optional so the
    Mac-local synthetic benchmark never fails because LIBERO is missing.
    """

    def __init__(self, hdf5_path=None, horizon: int = 16, suite: str = "libero_object"):
        if not libero_available():
            warnings.warn(
                "LIBERO is not installed. Synthetic benchmark remains available; "
                "install LIBERO only when running --dataset libero.",
                RuntimeWarning,
                stacklevel=2,
            )
        if hdf5_path is None:
            raise ImportError(
                "LIBERO native loading is optional and not configured here. "
                "Provide --data_path pointing at a converted LIBERO HDF5 file."
            )
        if not Path(hdf5_path).exists():
            raise FileNotFoundError(f"Converted LIBERO HDF5 not found: {hdf5_path}")
        self.suite = suite
        super().__init__(hdf5_path=hdf5_path, horizon=horizon, dataset_name="libero")

