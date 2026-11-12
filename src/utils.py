import copy, os, random, subprocess
import numpy as np
import torch
import yaml


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str) -> dict:
    """YAML with optional `base: other.yaml` inheritance (deep-merged), so ablations only list differences."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "base" in cfg:
        base = load_config(os.path.join(os.path.dirname(path), cfg.pop("base")))
        cfg = _merge(base, cfg)
    return cfg


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None


def save_ckpt(path, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)          # atomic: a crash mid-save cannot corrupt the checkpoint


def load_trainable(model, state_dict):
    """Copy saved trainable tensors (adapters or full weights) into the model."""
    params = dict(model.named_parameters())
    missing = [n for n in state_dict if n not in params]
    assert not missing, f"checkpoint keys not in model: {missing[:3]}"
    for n, v in state_dict.items():
        params[n].data.copy_(v)
