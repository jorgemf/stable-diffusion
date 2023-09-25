import os
from typing import Optional

import typer
import torch
from datasets import load_dataset, load_from_disk
from torchvision import transforms

from .clip import clip
from .config import IMAGE_SIZE, get_device

app = typer.Typer()


@app.command()
def download():
    print("Downloading 2M images...")
    datasets.load_dataset("poloclub/diffusiondb", "2m_all", num_proc=8)
    print("Done.")


@app.command()
def process():
    device = get_device()
    model, preprocess = clip.load("ViT-B/32", download_root='data/models', device=device)

    def _prompts_to_tokens(examples):
        with torch.no_grad():
            text = clip.tokenize(examples["prompt"], truncate=True).to(device)
            text_features = model.encode_text(text)
        examples["encoded_text"] = text_features.cpu()
        return examples

    ds = load_dataset("poloclub/diffusiondb", "2m_all", num_proc=8)['train']
    print("creating tokens from prompts...")
    ds = ds.map(_prompts_to_tokens,
                remove_columns=["seed", "step", "cfg", "sampler", "user_name", "timestamp",
                                "width", "height", "image_nsfw", "prompt_nsfw"],
                batched=True, batch_size=64)
    ds.save_to_disk("data/datasets/image_tokens")


def get_dataloader(transforms, batch_size: int, shuffle: bool, samples: Optional[int] = None) \
        -> torch.utils.data.DataLoader:
    ds = load_from_disk("data/datasets/image_tokens")
    if samples is not None:
        ds = ds.select(range(samples))
    ds = ds.with_format("torch")
    ds.set_transform(transforms)
    return torch.utils.data.DataLoader(ds,
                                       batch_size=batch_size,
                                       shuffle=shuffle,
                                       drop_last=False,
                                       pin_memory=True,
                                       num_workers=os.cpu_count())


def convert_image_to_rgb(image):
    return image.convert("RGB")


def get_dataset(batch_size: int,
                train: bool = True) -> torch.utils.data.DataLoader:
    operators = [
        transforms.Resize([IMAGE_SIZE]),
        transforms.CenterCrop(IMAGE_SIZE)
    ]
    if train:
        operators += [
            transforms.RandomHorizontalFlip(p=0.2)
        ]
    operators += [
        convert_image_to_rgb,
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ]
    data_transforms = transforms.Compose(operators)

    def _transforms(examples):
        examples["image"] = [data_transforms(x.convert("RGB")) for x in examples["image"]]
        examples["encoded_text"] = [torch.Tensor(x) for x in examples["encoded_text"]]
        return examples

    return get_dataloader(_transforms, batch_size, train)
