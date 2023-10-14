# HMA for Vision Transformers Finetuning

Code for fine-tuning ViT models on various classification datasets. Modified from [lucidrains/vit-pytorch](https://github.com/bwconrad/vit-finetune), basic hyperparameters follow [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929).

## Requirements

- Python 3.8+
- `pip install -r requirements.txt`

## Usage

### Training

To reproduce the full and part finetuning results in the paper, run the following commands:

```bash
python train.py --HMA  --max_steps 1000 --warmup_step 10 --lr 0.01  --seed 0  --tune_head --dataset dtd  --global_memory_number 16 --local_mem_size 512

python train.py --HMA  --max_steps 1000 --warmup_step 10 --lr 0.01  --seed 0  --tune_head --dataset dtd  --global_memory_number 16 --local_mem_size 512
```

#### Baselines

- To full and part fine-tune a ViT-B/16 model on dtd run:

```bash
python train.py  --max_steps 1000 --warmup_step 10 --lr 0.003  --seed 0 --dataset dtd

python train.py  --max_steps 1000 --warmup_step 10 --lr 0.01  --seed 0 --tune_head  --dataset dtd
```

- To add softpromts follow [Fine-tuning Image Transformers using Learnable Memory, CVPR2022](https://arxiv.org/abs/2203.15243), where `--mem_type` controls the type of prompts, `prop` means only add prompts to the first layer, `standard` means add prompts to all layers:

```bash
python train.py  --max_steps 1000 --warmup_step 10 --lr 0.003  --seed 0 --dataset dtd --mem_type prop/standard
```

## Demos

We provide attention scores demo of the **attention between datapoints (SMA)** module of tuned ViT in `./demo/atten_demo_dtd.ipynb`

## Available Datasets

| Dataset            | `--dataset` |
|:------------------:|:-----------:|
|[Describable Textures Dataset](https://www.robots.ox.ac.uk/~vgg/data/dtd/) | `dtd`|
|[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)| `cifar10`|
|[CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)| `cifar100`|
|[Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/)|  `pets37`|
|[Oxford Flowers-102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)|  `flowers102`|
|[Food-101](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/)|  `food101`|
|[STL-10](https://cs.stanford.edu/~acoates/stl10/)|  `stl10`|
|[Stanford Cars](https://ai.stanford.edu/~jkrause/cars/car_dataset.html) | `cars`|
|[FGVC Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/) | `aircraft`|
|[Image Folder](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html) | `custom`|
