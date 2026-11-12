import os
import sys


def set_environment():
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    # disable triton so torch uses the non-triton path
    sys.modules["triton"] = None
    sys.modules["triton.language"] = None


def set_seed(seed=42):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)


def mean_pool(hidden, mask):
    import torch

    mask = mask.unsqueeze(-1).expand(hidden.size()).float()
    summed = torch.sum(hidden * mask, 1)
    counts = torch.clamp(mask.sum(1), min=1e-9)
    return summed / counts
