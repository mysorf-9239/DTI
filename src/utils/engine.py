import os
import random
import csv
import numpy as np
import torch
from loguru import logger

NEG_LABEL = 0
POS_LABEL = 1

def to_device(x, device):
    """Move tensors/lists/dicts to the specified device."""
    if torch.is_tensor(x):
        return x.to(device, non_blocking=True)
    if isinstance(x, (list, tuple)):
        return type(x)(to_device(t, device) for t in x)
    if isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    return x

def setup_seed(seed: int):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Reproducibility seed set to: {seed}")

def check_dir(path: str):
    """Ensure a directory exists."""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
        logger.info(f"Created directory: {path}")

def csv_record(path: str, data: dict):
    """Record metrics/logs into a CSV file."""
    all_header = [
        "epoch", "batch", "lr", "loss", "avg_loss", "epoch_loss",
        "auprc", "auroc", "accuracy", "f1", "precision", "recall",
        "sensitivity", "specificity"
    ]
    row, header = [], []
    for k in all_header:
        if k in data:
            header.append(k)
            row.append(data[k])

    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if not file_exists:
            w.writerow(header)
        w.writerow(row)
