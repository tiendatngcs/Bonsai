"""Train a CIFAR CNN, optionally using Bonsai rematerialization."""

import argparse

import torch
from torch import nn
from tqdm.auto import tqdm

from example.models import CNN_MODELS
from example.utils import (
    create_cnn_model,
    get_cifar_loaders,
    get_device,
    unwrap_model_output,
    wrap_with_bonsai,
    wrap_with_rockmate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=sorted(CNN_MODELS), default="resnet18")
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar100")
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--budget-gb", type=float)
    parser.add_argument(
        "--model-weights-mb",
        type=float,
        help="Model weights memory in MB for schedule construction",
    )
    parser.add_argument("--scheduler", choices=("bonsai", "rockmate"), default="bonsai")
    parser.add_argument("--schedule-file", help="Cached schedule file under ./schedules")
    parser.add_argument("--trace-dir", default="./traces")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--trace-file-name", default=None, help="Custom trace file name. If not provided, defaults to operator_trace_<model_name>.txt")
    return parser.parse_args()


def evaluate(model: nn.Module, loader: torch.utils.data.DataLoader, device: torch.device) -> float:
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in loader:
            logits = unwrap_model_output(model(images.to(device)))
            predictions = logits.argmax(dim=1)
            correct += (predictions.cpu() == labels).sum().item()
            total += labels.numel()
    return correct / total


def main() -> None:
    args = parse_args()
    if args.max_steps is not None and args.max_steps < 1:
        raise ValueError("--max-steps must be positive")
    if args.model_weights_mb is not None and args.model_weights_mb < 0:
        raise ValueError("--model-weights-mb must be non-negative")
    device = get_device(args.device)
    train_loader, test_loader, num_classes = get_cifar_loaders(
        args.dataset, args.data_dir, args.batch_size, args.workers
    )
    model = create_cnn_model(args.model, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    sample_images, sample_labels = next(iter(train_loader))
    sample_images, sample_labels = sample_images.to(device), sample_labels.to(device)
    model_name = f"{args.model}_{args.dataset}"
    print(f"Training {args.model} on {args.dataset} with batch size {args.batch_size} for {args.epochs} epochs on device {device}")
    if args.scheduler == "rockmate":
        model = wrap_with_rockmate(
            model,
            sample_images,
            args.budget_gb,
            model_name,
            schedule_file=args.schedule_file,
        )
    else:
        model = wrap_with_bonsai(
            model,
            sample_images,
            lambda output: criterion(unwrap_model_output(output), sample_labels),
            args.budget_gb,
            trace_dir=args.trace_dir,
            model_name=model_name,
            trace_file_name=args.trace_file_name,
            schedule_file=args.schedule_file,
            weight_mb=args.model_weights_mb,
        )
    optimizer = torch.optim.SGD(
        model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=5e-4
    )

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        max_steps = (
            len(train_loader)
            if args.max_steps is None
            else min(len(train_loader), args.max_steps)
        )
        progress_bar = tqdm(
            train_loader,
            total=max_steps,
            desc=f"Epoch {epoch}/{args.epochs}",
            unit="batch",
        )
        for steps, (images, labels) in enumerate(progress_bar, start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(unwrap_model_output(model(images)), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            if args.max_steps is not None and steps >= args.max_steps:
                break
        accuracy = evaluate(model, test_loader, device)
        print(f"epoch={epoch} loss={total_loss / steps:.4f} accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()
    
# usage
# python -m example.train_cnn --model resnet18 --dataset cifar100 --epochs 2 --budget 0.96 --trace-file-name operator_trace_resnet18_256_RTX6000.txt
