from typing import Tuple, Optional
from pathlib import Path
import os

import torch
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from pytorch_fid.fid_score import calculate_frechet_distance
from pytorch_fid.inception import InceptionV3
import numpy as np
from .config import IMAGE_SIZE
from .diffusion import GaussianDiffusion
from .clip import clip
from .dataset import get_dataloader, convert_image_to_rgb


def test():
    pass


def create_images(model: torch.nn.Module, encoded_texts: torch.Tensor,
                  diffusion: GaussianDiffusion) -> torch.Tensor:
    pass


def calculate_clip_score(images: torch.Tensor, encoded_texts: torch.Tensor,
                         clip_model: torch.nn.Module) -> float:
    with torch.no_grad():
        text = clip.tokenize(examples["prompt"], truncate=True).to(device)
        text_features = model.encode_text(text)
    pass


def save_dataset_stats(path: Path, m2: float, s2: float) -> None:
    path = os.path.join(path, "dataset_stats")
    np.savez_compressed(path, m2=m2, s2=s2)


def load_dataset_stats(path: Path) -> Tuple[Optional[float], Optional[float]]:
    path = os.path.join(path, "dataset_stats")
    try:
        ckpt = np.load(path + ".npz")
        if "m2" in ckpt and "s2" in ckpt:
            return ckpt["m2"], ckpt["s2"]
        ckpt.close()
    except:
        pass
    return None, None


def get_dataset_stats(path: Path, dataloader:torch.utils.data.DataLoader,
                      inception_model: InceptionV3, batch_size: int, device: torch.device) \
        -> Tuple[float, float]:
    m2, s2 = load_dataset_stats(path)
    if m2 is None or s2 is None:

        total_steps = len(dataloader) // batch_size
        clip_score = 0
        fid_gen_arr = np.empty((len(dataloader), inception_block_idx))

        fid_real_arr = np.empty((len(dataloader), inception_block_idx))
        progress_bar = tqdm(total=total_steps, leave=False, position=1)
        with torch.no_grad():
            for batch, data in enumerate(dataloader):
                progress_bar.set_description(f"Step {batch + 1}/{total_steps}")
                encoded_texts = data['encoded_text'].to(dtype).to(device)
                images = create_images(model, encoded_texts, diffusion)

                incep_pred = inceptionv3_model(data["image_inception"].to(device))[0]
                start_idx = batch * batch_size
                fid_real_arr[start_idx:start_idx + incep_pred.shape[0]] = incep_pred.cpu().numpy()
                incep_pred = inceptionv3_model(images.to(device))[0]
                fid_gen_arr[start_idx:start_idx + incep_pred.shape[0]] = incep_pred.cpu().numpy()

                clip_score += calculate_clip_score(clip_preprocess(images), encoded_texts,
                                                   clip_model)
                progress_bar.update(1)
                if batch == 0:
                    summary_writer.add_images("test/images", images, global_step=global_step)
                    summary_writer.add_text("test/text", data['prompt'], global_step=global_step)

        save_dataset_stats(path, m2, s2)
    return m2, s2


def test_epoch(diffusion: GaussianDiffusion,
               model: torch.nn.Module,
               batch_size: int,
               samples: int,
               summary_writer: SummaryWriter,
               device: torch.device,
               dtype: torch.dtype,
               epoch: int,
               global_step: int,
               inception_block_idx: int = 2048) -> float:
    model.eval()
    clip_model, clip_preprocess = clip.load("ViT-B/32",
                                            download_root='data/models',
                                            device=device)
    assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
    inceptionv3_model = InceptionV3([block_idx]).to(device).eval()

    image_transform = transforms.Compose([
        transforms.Resize([IMAGE_SIZE]),
        transforms.CenterCrop(IMAGE_SIZE),
        convert_image_to_rgb,
    ])

    def _transforms(examples):
        examples["encoded_text"] = [torch.Tensor(x) for x in examples["encoded_text"]]
        examples["image_inception"] = clip_preprocess(examples["image"])
        examples["image_clip"] = clip_preprocess(examples["image"])
        examples["image"] = image_transform(examples["image"])
        return examples

    dataloader = get_dataloader(_transforms, batch_size=batch_size, shuffle=False, samples=samples)
    m2, s2 = get_dataset_stats(path, dataloader, inceptionv3_model, batch_size, device)

    total_steps = len(dataloader) // batch_size
    clip_score = 0
    fid_gen_arr = np.empty((len(dataloader), inception_block_idx))

    fid_real_arr = np.empty((len(dataloader), inception_block_idx))
    progress_bar = tqdm(total=total_steps, leave=False, position=1)
    with torch.no_grad():
        for batch, data in enumerate(dataloader):
            progress_bar.set_description(f"Step {batch + 1}/{total_steps}")
            encoded_texts = data['encoded_text'].to(dtype).to(device)
            images = create_images(model, encoded_texts, diffusion)

            incep_pred = inceptionv3_model(data["image_inception"].to(device))[0]
            start_idx = batch * batch_size
            fid_real_arr[start_idx:start_idx + incep_pred.shape[0]] = incep_pred.cpu().numpy()
            incep_pred = inceptionv3_model(images.to(device))[0]
            fid_gen_arr[start_idx:start_idx + incep_pred.shape[0]] = incep_pred.cpu().numpy()

            clip_score += calculate_clip_score(clip_preprocess(images), encoded_texts, clip_model)
            progress_bar.update(1)
            if batch == 0:
                summary_writer.add_images("test/images", images, global_step=global_step)
                summary_writer.add_text("test/text", data['prompt'], global_step=global_step)

    fid_value = calculate_frechet_distance(np.mean(fid_real_arr, axis=0),
                                           np.cov(fid_real_arr, rowvar=False),
                                           np.mean(fid_gen_arr, axis=0),
                                           np.cov(fid_gen_arr, rowvar=False))
    clip_score /= total_steps
    summary_writer.add_scalar("test/fid_score", fid_score, global_step=global_step)
    summary_writer.add_scalar("test/clip_score", clip_score, global_step=global_step)
    print(f"FID score: {fid_score:>8f}")
    print(f"CLIP score: {clip_score:>8f}")
    return fid_score
