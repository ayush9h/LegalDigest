from pathlib import Path

import torch
from omegaconf import OmegaConf


def load_config(model: str):
    root = Path(__file__).resolve().parents[2]
    cfg_dir = root / "legal_digest" / "config"

    model_cfg = OmegaConf.load(cfg_dir / "model.yaml")
    training_cfg = OmegaConf.load(cfg_dir / "training.yaml")
    data_cfg = OmegaConf.load(cfg_dir / "data.yaml")

    cfg = OmegaConf.merge(model_cfg, training_cfg, data_cfg)

    if model not in ("flan_t5", "llama"):
        raise ValueError("model must be one of: flan_t5, llama")

    cfg.model.active = cfg.model[model]

    cfg.runtime = {}
    cfg.runtime.device = "cuda" if torch.cuda.is_available() else "cpu"  # type: ignore

    return cfg
