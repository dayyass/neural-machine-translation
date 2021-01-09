from collections import defaultdict
from typing import Callable, DefaultDict, List, Optional

import numpy as np
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
    train_eval_freq: int = 500,
    verbose: bool = True,
):
    """
    Training loop on one epoch.
    """

    metrics: DefaultDict[str, List[float]] = defaultdict(list)

    if verbose:
        dataloader = tqdm(dataloader)

    model.train()

    for i, (input_lang_seq, output_lang_seq) in enumerate(dataloader):
        input_lang_seq = input_lang_seq.to(device)
        output_lang_seq = output_lang_seq.to(device)

        inputs = (input_lang_seq, output_lang_seq[:, :-1])
        targets = output_lang_seq[:, 1:]

        # forward pass
        outputs = model(*inputs)
        loss = criterion(outputs.transpose(1, 2), targets)  # ignore index used

        # backward pass
        loss.backward()

        # clip grad norm
        grad_norm = nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=5,  # hardcoded
        )

        # optimizer step
        optimizer.step()
        optimizer.zero_grad()

        # make predictions
        y_true = to_numpy(targets).tolist()
        y_pred = to_numpy(outputs.argmax(dim=-1)).tolist()

        # calculate metrics
        metrics = calculate_metrics(
            metrics=metrics,
            loss=loss.item(),
            grad_norm=grad_norm.item(),
            y_true=y_true,
            y_pred=y_pred,
        )

        if verbose:
            if i % train_eval_freq == 0:
                for metric_name, metric_list in metrics.items():
                    print(f"{metric_name}: {np.mean(metric_list[-train_eval_freq:])}")
                print()

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

    for input_lang_seq, output_lang_seq in dataloader:
        input_lang_seq = input_lang_seq.to(device)
        output_lang_seq = output_lang_seq.to(device)

        inputs = (input_lang_seq, output_lang_seq[:, :-1])
        targets = output_lang_seq[:, 1:]

        # forward pass
        with torch.no_grad():
            outputs = model(*inputs)
            loss = criterion(outputs.transpose(1, 2), targets)

        # make predictions
        y_true = to_numpy(targets).tolist()
        y_pred = to_numpy(outputs.argmax(dim=-1)).tolist()

        # calculate metrics
        metrics = calculate_metrics(
            metrics=metrics,
            loss=loss.item(),
            y_true=y_true,
            y_pred=y_pred,
        )

    return metrics


def train(
    model: nn.Module,
    trainloader: DataLoader,
    valloader: DataLoader,
    criterion: Callable,
    optimizer: optim.Optimizer,
    device: torch.device,
    n_epoch: int,
    train_eval_freq: int = 500,
    testloader: Optional[DataLoader] = None,
    scheduler: Optional[optim.lr_scheduler._LRScheduler] = None,
    verbose: bool = True,
):
    """
    Training / validation loop for n_epoch with final testing.
    """

    for epoch in range(n_epoch):

        if verbose:
            print(f"epoch [{epoch+1}/{n_epoch}]\n")

        train_metrics = train_epoch(
            model=model,
            dataloader=trainloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train_eval_freq=train_eval_freq,
            verbose=verbose,
        )

        if verbose:
            for metric_name, metric_list in train_metrics.items():
                print(f"train {metric_name}: {np.mean(metric_list)}")
            print()

        val_metrics = validate_epoch(
            model=model,
            dataloader=valloader,
            criterion=criterion,
            device=device,
            verbose=verbose,
        )

        if verbose:
            for metric_name, metric_list in val_metrics.items():
                print(f"val {metric_name}: {np.mean(metric_list)}")
            print()

        if scheduler is not None:
            scheduler.step()

    if testloader is not None:

        test_metrics = validate_epoch(
            model=model,
            dataloader=testloader,
            criterion=criterion,
            device=device,
            verbose=verbose,
        )

        if verbose:
            for metric_name, metric_list in test_metrics.items():
                print(f"test {metric_name}: {np.mean(metric_list)}")
            print()
