from datetime import datetime
from tqdm.auto import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.functional import mse_loss
from torch.profiler import profile, ProfilerActivity
from diffusers.optimization import get_cosine_schedule_with_warmup
from .config import get_device, TIMESTEPS, SAMPLING_TIMESTEPS, IMAGE_SIZE, RUNS_DIR, DTYPE
from .dataset import get_dataset
from .model import UNet
from .test import test_epoch
from .diffusion import GaussianDiffusion


def train(batch_size: int = 16,
          epochs: int = 10,
          learning_rate: float = 1e-4,
          test_samples: int = 30000,
          profiler=False,
          experiment_dir=None) -> None:
    dtype = DTYPE
    if experiment_dir is None:
        experiment_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir = RUNS_DIR.joinpath(experiment_dir)
    writer = SummaryWriter(experiment_dir.joinpath("logs/train"))
    train_dataloader = get_dataset(batch_size=batch_size, train=True)

    device = get_device()
    model = UNet(IMAGE_SIZE).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=len(train_dataloader) // 2,
        num_training_steps=(len(train_dataloader) * epochs),
    )

    model = model.train()
    diffusion = GaussianDiffusion(TIMESTEPS, SAMPLING_TIMESTEPS, dtype=dtype, device=device)
    progress_bar = tqdm(total=epochs,leave=True, position=0)

    def _train_loop():
        size_dataset = len(train_dataloader.dataset)
        best_score = test_epoch(diffusion, model, batch_size, test_samples, writer,
                           device=device, dtype=dtype, epoch=0, global_step=0)
        for e in range(epochs):
            progress_bar.set_description(f"Epoch {e + 1}/{epochs}")
            epoch(train_dataloader, diffusion, e, model, optimizer, lr_scheduler, writer,
                  device=device, dtype=dtype)
            progress_bar.update(1)
            score = test_epoch(diffusion, model, batch_size, test_samples, writer,
                               device=device, dtype=dtype, epoch=e, global_step=(e+1)*size_dataset)
            if (best_score is None) or (score < best_score):
                best_score = score
                print(f"New best score: {best_score}")
                torch.save(model.state_dict(), experiment_dir.joinpath(f"model_{e+1}.pt"))
            model.train()

    def _trace_handler(prof):
        output = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
        print(output)
        prof.export_chrome_trace(experiment_dir.joinpath("trace_" + str(prof.step_num) + ".json"))

    if profiler:
        with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=torch.profiler.schedule(skip_first=10, wait=10, warmup=5, active=2),
                on_trace_ready=_trace_handler
        ):
            _train_loop()
    else:
        _train_loop()
    writer.close()


def epoch(dataloader: torch.utils.data.DataLoader,
          diffusion: GaussianDiffusion,
          current_epoch: int,
          model: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
          summary_writer: SummaryWriter,
          device: torch.device,
          dtype: torch.dtype) -> None:
    size = len(dataloader.dataset)

    total_steps = len(dataloader)
    progress_bar = tqdm(total=total_steps, leave=False, position=1)
    for batch, data in enumerate(dataloader):
        progress_bar.set_description(f"Step {batch + 1}/{total_steps}")
        images = data['image'].to(dtype).to(device)
        encoded_texts = data['encoded_text'].to(dtype).to(device)
        batch_size = len(images)

        # Sample noise to add to the images
        noise = torch.randn(images.shape, dtype=images.dtype, device=images.device)
        # Sample a random timestep for each image
        timesteps = torch.randint(0, TIMESTEPS, (batch_size,), device=images.device).long()
        # Predict the noise residual
        x = diffusion.q_sample(image=images, noise=noise, t=timesteps)

        # For using a bigger batch size that the GPU supports:
        # https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation
        # https://huggingface.co/docs/diffusers/main/en/tutorials/basic_training

        noise_pred = model(x, timesteps, encoded_texts)
        noise_pred = torch.clamp(noise_pred, min=-1., max=1.)

        loss = mse_loss(noise_pred, noise)

        # Backpropagation
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        # Logging
        step = (batch + 1) * batch_size + size * current_epoch
        summary_writer.add_scalar('loss', loss.item(), global_step=step)

        # update progress bar
        progress_bar.update(1)
        logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
        progress_bar.set_postfix(**logs)

        if batch % 10 == 0:
            summary_writer.add_scalar('loss', loss.item(), global_step=step)
