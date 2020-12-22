from collections import defaultdict
from typing import Callable, DefaultDict, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from metrics import calculate_metrics
from utils import to_numpy


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

    metrics: DefaultDict[str, List[float]] = defaultdict(list)

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

        # make predictions
        # batch_size = 1 hardcoded
        y_true = to_numpy(targets)[0].tolist()
        y_pred = to_numpy(outputs.argmax(dim=-1))[0].tolist()

        # calculate metrics
        metrics = calculate_metrics(
            metrics=metrics,
            loss=loss.item(),
            y_true=y_true,
            y_pred=y_pred,
        )

    return metrics


def validate_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: Callable,
    device: torch.device,
    verbose: bool = True,
):
    """
    Validate loop on one epoch.
    """

    metrics: DefaultDict[str, List[float]] = defaultdict(list)

    if verbose:
        dataloader = tqdm(dataloader)

    model.eval()

    for from_seq, to_seq in dataloader:
        from_seq, to_seq = from_seq.to(device), to_seq.to(device)

        inputs = (from_seq, to_seq[:, :-1])
        targets = to_seq[:, 1:]

        # forward pass
        with torch.no_grad():
            outputs = model(*inputs)
            loss = criterion(outputs.transpose(1, 2), targets)

        # make predictions
        # batch_size = 1 hardcoded
        y_true = to_numpy(targets)[0].tolist()
        y_pred = to_numpy(outputs.argmax(dim=-1))[0].tolist()

        # calculate metrics
        metrics = calculate_metrics(
            metrics=metrics,
            loss=loss.item(),
            y_true=y_true,
            y_pred=y_pred,
        )

    return metrics
