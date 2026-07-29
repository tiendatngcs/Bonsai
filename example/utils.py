"""Shared model, data, and Bonsai helpers."""

from collections.abc import Callable
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from example.models import CNN_MODELS


def get_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested_device)


def normalize_model_name(model_name: str) -> str:
    return model_name.replace("-", "_").replace("/", "_")


def get_cached_schedule_file(schedule_file: str | None) -> str | None:
    if schedule_file is None:
        return None

    schedules_dir = Path("schedules").resolve()
    resolved_schedule_file = Path(schedule_file).resolve()
    if not resolved_schedule_file.is_relative_to(schedules_dir):
        raise ValueError("--schedule-file must reference a file under ./schedules")
    if not resolved_schedule_file.is_file():
        raise FileNotFoundError(f"Cached schedule file not found: {resolved_schedule_file}")
    return str(resolved_schedule_file)


def create_cnn_model(name: str, num_classes: int) -> nn.Module:
    try:
        factory: Callable[[int], nn.Module] = CNN_MODELS[name]
    except KeyError as error:
        supported = ", ".join(sorted(CNN_MODELS))
        raise ValueError(f"Unsupported CNN model {name!r}. Choose one of: {supported}") from error
    return factory(num_classes)


def get_cifar_loaders(
    dataset_name: str,
    data_dir: str,
    batch_size: int,
    workers: int,
) -> tuple[DataLoader, DataLoader, int]:
    dataset_classes = {
        "cifar10": (datasets.CIFAR10, 10, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        "cifar100": (datasets.CIFAR100, 100, (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    }
    try:
        dataset_class, num_classes, mean, std = dataset_classes[dataset_name]
    except KeyError as error:
        raise ValueError("dataset_name must be 'cifar10' or 'cifar100'") from error

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    train_dataset = dataset_class(data_dir, train=True, download=True, transform=train_transform)
    test_dataset = dataset_class(data_dir, train=False, download=True, transform=test_transform)
    loader_options = {"num_workers": workers, "pin_memory": torch.cuda.is_available()}
    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **loader_options),
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_options),
        num_classes,
    )


def wrap_with_bonsai(
    model: nn.Module,
    sample: torch.Tensor,
    loss_fn: Callable[[torch.Tensor], torch.Tensor],
    budget_gb: float | None,
    trace_dir: str,
    model_name: str,
    trace_file_name: str | None = None,
    schedule_file: str | None = None,
    weight_mb: float | None = None,
) -> nn.Module:
    model_name = normalize_model_name(model_name)
    schedule_file = get_cached_schedule_file(schedule_file)
    if budget_gb is None:
        print(f"Not wrapping model {model_name} with Bonsai (no budget specified)")
        return model

    from rockmate import Bonsai

    model.model_name = model_name
    return Bonsai(
        model=model,
        inputs=sample,
        budget_GB=budget_gb,
        trace_dir=trace_dir,
        model_name=model_name,
        loss_fn=loss_fn,
        trace_file_name=trace_file_name,
        schedule_file=schedule_file,
        weight_MB=weight_mb,
    )


def wrap_with_rockmate(
    model: nn.Module,
    sample: torch.Tensor,
    budget_gb: float | None,
    model_name: str,
    schedule_file: str | None = None,
) -> nn.Module:
    model_name = normalize_model_name(model_name)
    schedule_file = get_cached_schedule_file(schedule_file)
    if budget_gb is None:
        print(f"Not wrapping model {model_name} with Rockmate (no budget specified)")
        return model

    from rockmate import RockmateFunc

    model.model_name = model_name
    return RockmateFunc(
        model=model,
        inputs=sample,
        budget_GB=budget_gb,
        schedule_file=schedule_file,
    )


def unwrap_model_output(output: torch.Tensor | tuple[torch.Tensor, ...]) -> torch.Tensor:
    return output[0] if isinstance(output, tuple) else output
