# Quick Train Guide


Training with different models and budget types using cached schedules.

## Step 1 - Decompress the cached recomputation schedules.

```bash
cd ./schedules

sudo apt update
sudo apt install git-lfs
git lfs install
git lfs pull

bash unzip_model_schedules.sh

cd ..
```

## Step 2 - Pull and run the docker image

Use the Docker image to run workloads against cached Rockmate, Checkmate, or Bonsai (re)computation schedules without building PyTorch or solving for a new (re)computation schedules. Docker and the NVIDIA Container Toolkit are required for GPU execution.

```bash
docker pull tiendatngcs/bonsai-exec-only:dev
```

Run the docker image with

```bash
IMAGE_NAME=tiendatngcs/bonsai-exec-only:dev source ./docker/run_container.sh
```

## Step 3 - Train models with cached recomputation schedules

### Commands

The following commands train models with configurations used in the evaluation section in our paper.


| Model | Small | Medium | Large |
|---|---|---|---|
| ResNet-18 (batch 256) | [Rockmate](example/commands/from_cache_schedule/train_rockmate_resnet18_small.sh)<br>[Bonsai](example/commands/from_cache_schedule/train_bonsai_resnet18_small.sh) | [Rockmate](example/commands/from_cache_schedule/train_rockmate_resnet18_medium.sh)<br>[Bonsai](example/commands/from_cache_schedule/train_bonsai_resnet18_medium.sh) | [Rockmate](example/commands/from_cache_schedule/train_rockmate_resnet18_large.sh)<br>[Bonsai](example/commands/from_cache_schedule/train_bonsai_resnet18_large.sh) |
| ResNet-50 (batch 256) | [Rockmate](example/commands/from_cache_schedule/train_rockmate_resnet50_small.sh)<br>[Bonsai](example/commands/from_cache_schedule/train_bonsai_resnet50_small.sh) | [Rockmate](example/commands/from_cache_schedule/train_rockmate_resnet50_medium.sh)<br>[Bonsai](example/commands/from_cache_schedule/train_bonsai_resnet50_medium.sh) | [Rockmate](example/commands/from_cache_schedule/train_rockmate_resnet50_large.sh)<br>[Bonsai](example/commands/from_cache_schedule/train_bonsai_resnet50_large.sh) |
| GoogLeNet (batch 256) | [Rockmate](example/commands/from_cache_schedule/train_rockmate_googlenet_small.sh)<br>[Bonsai](example/commands/from_cache_schedule/train_bonsai_googlenet_small.sh) | [Rockmate](example/commands/from_cache_schedule/train_rockmate_googlenet_medium.sh)<br>[Bonsai](example/commands/from_cache_schedule/train_bonsai_googlenet_medium.sh) | [Rockmate](example/commands/from_cache_schedule/train_rockmate_googlenet_large.sh)<br>[Bonsai](example/commands/from_cache_schedule/train_bonsai_googlenet_large.sh) |
| Inception-v3 (batch 256) | [Rockmate](example/commands/from_cache_schedule/train_rockmate_inceptionv3_small.sh)<br>[Bonsai](example/commands/from_cache_schedule/train_bonsai_inceptionv3_small.sh) | [Rockmate](example/commands/from_cache_schedule/train_rockmate_inceptionv3_medium.sh)<br>[Bonsai](example/commands/from_cache_schedule/train_bonsai_inceptionv3_medium.sh) | [Rockmate](example/commands/from_cache_schedule/train_rockmate_inceptionv3_large.sh)<br>[Bonsai](example/commands/from_cache_schedule/train_bonsai_inceptionv3_large.sh) |
| ResNet-152 (batch 256) | [Rockmate](example/commands/from_cache_schedule/train_rockmate_resnet152_small.sh)<br>[Bonsai](example/commands/from_cache_schedule/train_bonsai_resnet152_small.sh) | [Rockmate](example/commands/from_cache_schedule/train_rockmate_resnet152_medium.sh)<br>[Bonsai](example/commands/from_cache_schedule/train_bonsai_resnet152_medium.sh) | [Rockmate](example/commands/from_cache_schedule/train_rockmate_resnet152_large.sh)<br>[Bonsai](example/commands/from_cache_schedule/train_bonsai_resnet152_large.sh) |
| EleutherAI/pythia-160m (batch 16) | [Rockmate](example/commands/from_cache_schedule/train_rockmate_pythia_160m_small.sh)<br>[Bonsai](example/commands/from_cache_schedule/train_bonsai_pythia_160m_small.sh) | [Rockmate](example/commands/from_cache_schedule/train_rockmate_pythia_160m_medium.sh)<br>[Bonsai](example/commands/from_cache_schedule/train_bonsai_pythia_160m_medium.sh) | [Rockmate](example/commands/from_cache_schedule/train_rockmate_pythia_160m_large.sh)<br>[Bonsai](example/commands/from_cache_schedule/train_bonsai_pythia_160m_large.sh) |
| openai-community/gpt2 (batch 8) | [Rockmate](example/commands/from_cache_schedule/train_rockmate_gpt2_small.sh)<br>[Bonsai](example/commands/from_cache_schedule/train_bonsai_gpt2_small.sh) | [Rockmate](example/commands/from_cache_schedule/train_rockmate_gpt2_medium.sh)<br>[Bonsai](example/commands/from_cache_schedule/train_bonsai_gpt2_medium.sh) | [Rockmate](example/commands/from_cache_schedule/train_rockmate_gpt2_large.sh)<br>[Bonsai](example/commands/from_cache_schedule/train_bonsai_gpt2_large.sh) |
| facebook/opt-350m (batch 8) | [Rockmate](example/commands/from_cache_schedule/train_rockmate_opt_350m_small.sh)<br>[Bonsai](example/commands/from_cache_schedule/train_bonsai_opt_350m_small.sh) | [Rockmate](example/commands/from_cache_schedule/train_rockmate_opt_350m_medium.sh)<br>[Bonsai](example/commands/from_cache_schedule/train_bonsai_opt_350m_medium.sh) | [Rockmate](example/commands/from_cache_schedule/train_rockmate_opt_350m_large.sh)<br>[Bonsai](example/commands/from_cache_schedule/train_bonsai_opt_350m_large.sh) |
