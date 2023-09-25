import datasets
from pathlib import Path
import torch

IMAGE_SIZE = 64
TIMESTEPS = 500
SAMPLING_TIMESTEPS = 100
RUNS_DIR = Path("./runs")
DTYPE = torch.bfloat16


def setup():
    print("Setting up config...")
    datasets.config.DOWNLOADED_DATASETS_PATH = Path("./data/datasets")
    datasets.config.HF_DATASETS_CACHE = Path("./data/cache")

    device = get_device()
    print(f"Using {device} device")

    RUNS_DIR.mkdir(exist_ok=True)


def get_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return device
