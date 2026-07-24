# Full workflow

Both evaluation on training and solving (using GUROBI) can be done in a Anaconda virutal environment.  

## Installation

This setup builds PyTorch from source with the Bonsai batch patch applied, installs TorchVision,
and then installs Rockmate. The `pytorch/` and `vision/` directories are **not** part of this
repository — they are cloned automatically by the scripts below.


### Step 0a — (prerequisite) Install the Ananconda

Official Anaconda installation guide can be found here.

https://www.anaconda.com/docs/getting-started/anaconda/install/linux-install

### Step 0b (prerequisite) — Install the GUROBI solver

Both Bonsai and Rockmate use GUROBI for ILP solving, so install GUROBI and make sure the `gurobi_cl`
command is available on your `PATH` before continuing.

Follow the official installation guide for your platform:

https://www.gurobi.com/documentation/


You should be able to attain a valid *Named-User Academic* licence if you are a student, staff, or faculty.**.
*Please feel free to let us know if you do not have access to a valid GUROBI licence.*

---

### Step 1 — Create the Conda environment and install customized PyTorch (for tracer features)

The Bonsai model tracer is integrated into a customized PyTorch build.

Running the following script will automatically:
- create a Conda environment
- clone PyTorch, check out the correct version, apply the Bonsai patch, build it, and install it

```bash
source create_default_pytorch_env.sh
```

---

### Step 2 — Install Rockmate-Bonsai

Bonsai is implemented as a module in [`Rockmate`](https://github.com/topal-team/rockmate) to use Rockmate's scheduling infrastructure.

```bash
pip install -e ./rockmate-bonsai/rkgb -e ./rockmate-bonsai/rockmate
```

---

**From here, please verify that customized PyTorch and Torch vision is installed**

```bash
pip show torch torchvision
```

Outputs:
```
Name: torch
Version: 2.3.0a0
Summary: Tensors and Dynamic neural networks in Python with strong GPU acceleration
Home-page: https://pytorch.org/
Author: PyTorch Team
Author-email: packages@pytorch.org
License: BSD-3
Location: /home/cc/anaconda3/envs/pytorch_env_bonsai/lib/python3.11/site-packages
Editable project location: /home/cc/pytorch
Requires: filelock, fsspec, jinja2, networkx, sympy, typing-extensions
Required-by: rkgb, rockmate, torchvision
---
Name: torchvision
Version: 0.18.0a0+6043bc2
Summary: image and video datasets and models for torch deep learning
Home-page: https://github.com/pytorch/vision
Author: PyTorch Core Team
Author-email: soumith@pytorch.org
License: BSD
Location: /home/cc/anaconda3/envs/pytorch_env_bonsai/lib/python3.11/site-packages
Editable project location: /home/cc/vision
Requires: numpy, pillow, torch
```

### Version Reference

Versions used in the Bonsai configuration:

| Component    | Version |
|--------------|---------|
| Gurobi       | 12.0.3  |
| Python       | 3.11    |
| PyTorch      | 2.3.0 (customized)   |
| TorchVision  | 0.18.0 (from source) |
| Transformers | 4.36.0  |

## Example commands

### CNNs

Train a CNN model with CIFAR100 dataset:

```bash
python -m example.train_cnn --model resnet18 --dataset cifar100 --epochs 2
```

Add `--budget-gb 8` to construct the model with Bonsai. The command downloads
the selected CIFAR dataset to `./data` when necessary.

### Transformers

Run a transformer model:

```bash
python -m example.train_transformer --model EleutherAI/pythia-160m --steps 10
```

Set `--model` to any Hugging Face causal language model, and use `--budget-gb`
to enable Bonsai. The entry point trains on
[`databricks/databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k)
by default. Use `--dataset`, `--max-samples`, `--sequence-length`, and
`--steps` to configure the training data and run length.

### Commands

The following commands train models with configurations used in the evaluation section in our paper. 


| Model | Small | Medium | Large |
|---|---|---|---|
| ResNet-18 (batch 256) | [Rockmate](example/commands/solve_train/train_rockmate_resnet18_small.sh)<br>[Bonsai](example/commands/solve_train/train_bonsai_resnet18_small.sh) | [Rockmate](example/commands/solve_train/train_rockmate_resnet18_medium.sh)<br>[Bonsai](example/commands/solve_train/train_bonsai_resnet18_medium.sh) | [Rockmate](example/commands/solve_train/train_rockmate_resnet18_large.sh)<br>[Bonsai](example/commands/solve_train/train_bonsai_resnet18_large.sh) |
| ResNet-50 (batch 256) | [Rockmate](example/commands/solve_train/train_rockmate_resnet50_small.sh)<br>[Bonsai](example/commands/solve_train/train_bonsai_resnet50_small.sh) | [Rockmate](example/commands/solve_train/train_rockmate_resnet50_medium.sh)<br>[Bonsai](example/commands/solve_train/train_bonsai_resnet50_medium.sh) | [Rockmate](example/commands/solve_train/train_rockmate_resnet50_large.sh)<br>[Bonsai](example/commands/solve_train/train_bonsai_resnet50_large.sh) |
| GoogLeNet (batch 256) | [Rockmate](example/commands/solve_train/train_rockmate_googlenet_small.sh)<br>[Bonsai](example/commands/solve_train/train_bonsai_googlenet_small.sh) | [Rockmate](example/commands/solve_train/train_rockmate_googlenet_medium.sh)<br>[Bonsai](example/commands/solve_train/train_bonsai_googlenet_medium.sh) | [Rockmate](example/commands/solve_train/train_rockmate_googlenet_large.sh)<br>[Bonsai](example/commands/solve_train/train_bonsai_googlenet_large.sh) |
| Inception-v3 (batch 256) | [Rockmate](example/commands/solve_train/train_rockmate_inceptionv3_small.sh)<br>[Bonsai](example/commands/solve_train/train_bonsai_inceptionv3_small.sh) | [Rockmate](example/commands/solve_train/train_rockmate_inceptionv3_medium.sh)<br>[Bonsai](example/commands/solve_train/train_bonsai_inceptionv3_medium.sh) | [Rockmate](example/commands/solve_train/train_rockmate_inceptionv3_large.sh)<br>[Bonsai](example/commands/solve_train/train_bonsai_inceptionv3_large.sh) |
| ResNet-152 (batch 256) | [Rockmate](example/commands/solve_train/train_rockmate_resnet152_small.sh)<br>[Bonsai](example/commands/solve_train/train_bonsai_resnet152_small.sh) | [Rockmate](example/commands/solve_train/train_rockmate_resnet152_medium.sh)<br>[Bonsai](example/commands/solve_train/train_bonsai_resnet152_medium.sh) | [Rockmate](example/commands/solve_train/train_rockmate_resnet152_large.sh)<br>[Bonsai](example/commands/solve_train/train_bonsai_resnet152_large.sh) |
| EleutherAI/pythia-160m (batch 16) | [Rockmate](example/commands/solve_train/train_rockmate_pythia_160m_small.sh)<br>[Bonsai](example/commands/solve_train/train_bonsai_pythia_160m_small.sh) | [Rockmate](example/commands/solve_train/train_rockmate_pythia_160m_medium.sh)<br>[Bonsai](example/commands/solve_train/train_bonsai_pythia_160m_medium.sh) | [Rockmate](example/commands/solve_train/train_rockmate_pythia_160m_large.sh)<br>[Bonsai](example/commands/solve_train/train_bonsai_pythia_160m_large.sh) |
| openai-community/gpt2 (batch 8) | [Rockmate](example/commands/solve_train/train_rockmate_gpt2_small.sh)<br>[Bonsai](example/commands/solve_train/train_bonsai_gpt2_small.sh) | [Rockmate](example/commands/solve_train/train_rockmate_gpt2_medium.sh)<br>[Bonsai](example/commands/solve_train/train_bonsai_gpt2_medium.sh) | [Rockmate](example/commands/solve_train/train_rockmate_gpt2_large.sh)<br>[Bonsai](example/commands/solve_train/train_bonsai_gpt2_large.sh) |
| facebook/opt-350m (batch 8) | [Rockmate](example/commands/solve_train/train_rockmate_opt_350m_small.sh)<br>[Bonsai](example/commands/solve_train/train_bonsai_opt_350m_small.sh) | [Rockmate](example/commands/solve_train/train_rockmate_opt_350m_medium.sh)<br>[Bonsai](example/commands/solve_train/train_bonsai_opt_350m_medium.sh) | [Rockmate](example/commands/solve_train/train_rockmate_opt_350m_large.sh)<br>[Bonsai](example/commands/solve_train/train_bonsai_opt_350m_large.sh) |

## API usage

An example of Bonsai training can be found in [`example_train.py`](./example_train.py).
An example of how to generate a trace file can be found in [`example_tracer.py`](./example_tracer.py).

Reusable CNN and transformer model definitions, plus their training entry points,
are organized under [`example/`](./example/).


