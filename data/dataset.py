import os
import glob
import torch
from torch.utils.data import Dataset
import numpy as np

def _load_pt(path: str):
    return torch.load(path, map_location="cpu")

class MultiModalPTDataset(Dataset):
    def __init__(self, data_dir: str, paths: list[str]):
        self.data_dir = data_dir
        self.paths = paths

    @staticmethod
    def discover(data_dir: str):
        paths = sorted(glob.glob(os.path.join(data_dir, "*.pt")))
        if len(paths) == 0:
            raise FileNotFoundError(f"No .pt files found in {data_dir}")
        return paths

    @staticmethod
    def get_pid(path: str):
        obj = _load_pt(path)
        pid = obj.get("pid", None)
        if pid is None:
            base = os.path.basename(path)
            pid = base.split(".")[0]
        return str(pid)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        obj = _load_pt(self.paths[idx])

        wsi_latent = obj["wsi_latent"].float()
        wsi_lr = obj["wsi_lr"].float()
        genomics = obj["genomics"].float()

        y = obj["y"]
        if isinstance(y, torch.Tensor):
            y = int(y.item()) if y.numel() == 1 else y.long()
        else:
            y = int(y)

        time = obj["time"]
        censor = obj["censor"]
        if isinstance(time, torch.Tensor):
            time = float(time.item()) if time.numel() == 1 else float(time.mean().item())
        else:
            time = float(time)
        if isinstance(censor, torch.Tensor):
            censor = float(censor.item()) if censor.numel() == 1 else float(censor.mean().item())
        else:
            censor = float(censor)

        pid = obj.get("pid", None)
        if pid is None:
            pid = self.get_pid(self.paths[idx])
        pid = str(pid)

        return {
            "wsi_latent": wsi_latent,
            "wsi_lr": wsi_lr,
            "genomics": genomics,
            "y": torch.tensor(y, dtype=torch.long),
            "time": torch.tensor(time, dtype=torch.float32),
            "censor": torch.tensor(censor, dtype=torch.float32),
            "pid": pid,
        }

def make_patient_folds(paths: list[str], k: int, seed: int):
    pids = [MultiModalPTDataset.get_pid(p) for p in paths]
    uniq = sorted(list(set(pids)))
    rng = np.random.default_rng(seed)
    rng.shuffle(uniq)
    folds = [set() for _ in range(k)]
    for i, pid in enumerate(uniq):
        folds[i % k].add(pid)
    return folds

def split_by_fold(paths: list[str], fold_pids: set[str]):
    pids = [MultiModalPTDataset.get_pid(p) for p in paths]
    test = [p for p, pid in zip(paths, pids) if pid in fold_pids]
    train = [p for p, pid in zip(paths, pids) if pid not in fold_pids]
    return train, test