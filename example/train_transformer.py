"""Train a transformer model."""

import argparse
from itertools import islice

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from example.utils import (
    get_device,
    unwrap_model_output,
    wrap_with_bonsai,
    wrap_with_rockmate,
)


DATASET_ALIASES = {
    "databricks/dolly15k": "databricks/databricks-dolly-15k",
    "dolly-15k": "databricks/databricks-dolly-15k",
}


class TokenDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, texts: list[str], tokenizer: AutoTokenizer, sequence_length: int):
        encoded = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=sequence_length,
            return_tensors="pt",
        )
        self.encodings = encoded

    def __len__(self) -> int:
        return len(self.encodings["input_ids"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = {key: value[index] for key, value in self.encodings.items()}
        item["labels"] = item["input_ids"].clone()
        return item


# class ForwardWrapper(nn.Module):
#     def __init__(self, model: AutoModelForCausalLM):
#         super().__init__()
#         self.model = model

#     def forward(
#         self,
#         input_ids: torch.Tensor,
#         labels: torch.Tensor | None = None,
#     ) -> torch.Tensor:
#         return self.model(input_ids=input_ids, labels=labels).logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face causal language-model identifier")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=-1, help="Maximum training steps per epoch; -1 uses all batches")
    parser.add_argument("--dataset", default="databricks/databricks-dolly-15k")
    parser.add_argument("--max-samples", type=int, help="Limit the number of training examples")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
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
    parser.add_argument("--trace-file-name", default=None, help="Custom trace file name. If not provided, defaults to operator_trace_<model_name>.txt")
    return parser.parse_args()


def causal_loss(logits: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        token_ids[:, 1:].reshape(-1),
    )


def format_dolly_example(example: dict[str, str]) -> str:
    instruction = example.get("instruction") or ""
    response = example.get("response") or ""
    return f"Instruction: {instruction}\n\nResponse: {response}"


def create_dataset(args: argparse.Namespace, tokenizer: AutoTokenizer) -> TokenDataset:
    dataset_name = DATASET_ALIASES.get(args.dataset, args.dataset)
    dataset = load_dataset(dataset_name, split="train").train_test_split(
        test_size=0.1,
        seed=args.seed,
    )["train"]
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError("--max-samples must be positive")
        dataset = dataset.shuffle(seed=args.seed).select(
            range(min(args.max_samples, len(dataset)))
        )
    return TokenDataset(
        [format_dolly_example(example) for example in dataset],
        tokenizer,
        args.sequence_length,
    )


def train_loop(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    max_steps: int,
    log_interval: int,
) -> None:
    global_step = 0
    model.train()
    steps_before_final_batch = len(loader) - 1
    steps_per_epoch = (
        steps_before_final_batch
        if max_steps == -1
        else min(steps_before_final_batch, max_steps)
    )
    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        steps_completed = 0
        progress_bar = tqdm(
            islice(loader, steps_per_epoch),
            total=steps_per_epoch,
            desc=f"Epoch {epoch}/{epochs}",
            unit="batch",
        )
        for step, batch in enumerate(progress_bar, start=1):
            batch = {
                key: value.to(device, non_blocking=device.type == "cuda")
                for key, value in batch.items()
            }
            token_ids = batch["input_ids"]
            labels = token_ids
            # optimizer.zero_grad(set_to_none=True)
            outputs = model(token_ids, labels=labels)
            logits = unwrap_model_output(outputs)
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]
            loss = nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )
            loss.backward()
            optimizer.step()

            global_step += 1
            steps_completed += 1
            epoch_loss += loss.item()
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")
            # if step % log_interval == 0 or step == steps_per_epoch:
            #     print(
            #         f"epoch={epoch} step={step} global_step={global_step} "
            #         f"loss={loss.item():.4f}"
            #     )

        if steps_completed == 0:
            raise RuntimeError("The training loader did not yield any batches.")
        print(f"epoch={epoch} average_loss={epoch_loss / steps_completed:.4f}")


def main() -> None:
    args = parse_args()
    if args.sequence_length < 2:
        raise ValueError("--sequence-length must be at least 2 for causal language modeling")
    if args.epochs < 1 or args.steps < -1 or args.steps == 0:
        raise ValueError("--epochs must be positive and --steps must be -1 or positive")
    if args.log_interval < 1:
        raise ValueError("--log-interval must be positive")
    if args.model_weights_mb is not None and args.model_weights_mb < 0:
        raise ValueError("--model-weights-mb must be non-negative")

    torch.manual_seed(args.seed)
    device = get_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = create_dataset(args, tokenizer)
    if len(dataset) < args.batch_size:
        raise ValueError("The dataset must contain at least --batch-size examples")
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )
    sample = next(iter(loader))["input_ids"].to(device)
    model_name = args.model
    if args.scheduler == "rockmate":
        model = wrap_with_rockmate(
            model,
            sample,
            args.budget_gb,
            model_name,
            schedule_file=args.schedule_file,
        )
    else:
        model = wrap_with_bonsai(
            model,
            sample,
            lambda output: causal_loss(output, sample),
            args.budget_gb,
            trace_dir=args.trace_dir,
            model_name=model_name,
            trace_file_name=args.trace_file_name,
            schedule_file=args.schedule_file,
            weight_mb=args.model_weights_mb,
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    print(
        f"Training {args.model} for {args.epochs} epoch(s), up to {args.steps} "
        f"steps per epoch, on {device}"
    )
    train_loop(
        model,
        loader,
        optimizer,
        device,
        args.epochs,
        args.steps,
        args.log_interval,
    )


if __name__ == "__main__":
    main()


# usage
# python -m example.train_transformer --model gpt2 --batch-size 8 --sequence-length 64 --epochs 1 --steps 10 --budget-gb 0.96 --trace-file-name operator_trace_gpt2_64_RTX6000.txt