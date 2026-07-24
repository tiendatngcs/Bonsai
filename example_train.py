"""
Example: train ResNet-50 on CIFAR-100 (32x32 images) for two full epochs
using the Bonsai API.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from rockmate import Bonsai

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
batch_size = 100
train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
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

sample_images, sample_labels = next(iter(train_loader))
sample_images, sample_labels = sample_images.to(device), sample_labels.to(device)

# ---------------------------------------------------------------------------
# Bonsai – build a Bonsai-wrapped training model
# ---------------------------------------------------------------------------
budget_gb = 8
model = Bonsai(
    model=model,
    inputs=sample_images,                       # single tensor → model(images)
    budget_GB=budget_gb,
    trace_dir="./traces",
    model_name="resnet50_cifar100",
    loss_fn=lambda output: criterion(output, sample_labels),
)

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
    momentum=0.9,
    weight_decay=5e-4,
)

# ---------------------------------------------------------------------------
# Train the Bonsai-wrapped model for two epochs
# ---------------------------------------------------------------------------
num_epochs = 2
model.train()

for epoch in range(num_epochs):
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        logits = model(images)[0]             # Rockmate returns outputs as a tuple
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}/{num_epochs} - loss: {avg_loss:.4f}")

# usage: python example_train.py
