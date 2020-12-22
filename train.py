from typing import Callable

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable,
    optimizer: optim.Optimizer,
    device: torch.device,
    verbose: bool = True,
):
    """
    Training loop on one epoch.
    """

    if verbose:
        dataloader = tqdm(dataloader)

    model.train()

    for from_seq, to_seq in dataloader:
        from_seq, to_seq = from_seq.to(device), to_seq.to(device)

        inputs = (from_seq, to_seq[:, :-1])
        targets = to_seq[:, 1:]

        # forward pass
        outputs = model(*inputs)
        loss = criterion(outputs.transpose(1, 2), targets)

        # backward pass
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
