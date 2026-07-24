"""
Example: train ResNet-50 on CIFAR-100 (32x32 images) for one iteration
using BonsaiTracer.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from rockmate import BonsaiTracer

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------------------------------------------------------
# Dataset – CIFAR-100, images kept at native 32x32
# ---------------------------------------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.5071, 0.4867, 0.4408),
        std=(0.2675, 0.2565, 0.2761),
    ),
])

train_dataset = torchvision.datasets.CIFAR100(
    root="./data", train=True, download=True, transform=transform
)
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=2
)

# ---------------------------------------------------------------------------
# Model – ResNet-50 adapted for 32x32 inputs and 100 classes
# ---------------------------------------------------------------------------
model = torchvision.models.resnet50(weights=None)
# Replace the first conv layer: smaller kernel/stride so 32x32 features survive
model.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
model.maxpool = torch.nn.Identity()          # skip the aggressive spatial pooling
model.fc = torch.nn.Linear(model.fc.in_features, 100)  # 100 CIFAR-100 classes
model = model.to(device)

# ---------------------------------------------------------------------------
# Loss – cross-entropy with class labels
# ---------------------------------------------------------------------------
criterion = torch.nn.CrossEntropyLoss()

images, labels = next(iter(train_loader))
images, labels = images.to(device), labels.to(device)

# ---------------------------------------------------------------------------
# BonsaiTracer – one training iteration; traces are written to ./traces/
# ---------------------------------------------------------------------------
BonsaiTracer(
    model=model,
    inputs=images,                              # single tensor → model(images)
    trace_dir="./traces",
    model_name="resnet50_cifar100",
    loss_fn=lambda output: criterion(output, labels),
    optimizer_cls=torch.optim.SGD,
    optimizer_kwargs={"lr": 0.01, "momentum": 0.9, "weight_decay": 5e-4},
)

# usage: python example_tracer.py

