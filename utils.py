import random

import numpy as np
import torch


def set_global_seed(seed: int):
    """
    Set global seed for reproducibility.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert torch.Tensor to np.ndarray.
    """

    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def infer_length(tensor: torch.Tensor, pad_id):
    """
    PyTorch tensor get the index of specific value (list.index() analog).
    """

    assert tensor.dim() == 2
    lengths = torch.sum(tensor != pad_id, dim=-1)
    return lengths


def gather_hidden_states_by_length(tensor: torch.Tensor, lengths: torch.Tensor):
    """
    Gather appropriate hidden state by lengths (when use <PAD>).
    """

    assert tensor.dim() == 3
    index = torch.repeat_interleave(
        (lengths - 1).unsqueeze(-1).unsqueeze(-1), repeats=tensor.shape[-1], dim=-1
    )
    hidden = torch.gather(tensor, dim=1, index=index)
    return hidden
