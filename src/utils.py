import os, random
import numpy as np
import torch
import yaml


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def save_ckpt(path, model, optimizer, scheduler, scaler, epoch):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
    }, tmp)
    os.replace(tmp, path)  # atomic: a crash mid-save cannot corrupt the checkpoint


def load_ckpt_if_exists(path, model, optimizer, scheduler, scaler, device):
    """Returns the epoch to start from (0 if no checkpoint)."""
    if not os.path.exists(path):
        return 0
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck["model"])
    optimizer.load_state_dict(ck["optimizer"])
    scheduler.load_state_dict(ck["scheduler"])
    if scaler is not None and ck.get("scaler") is not None:
        scaler.load_state_dict(ck["scaler"])
    print(f"[resume] loaded {path}, continuing from epoch {ck['epoch'] + 1}")
    return ck["epoch"] + 1
