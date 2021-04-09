from pathlib import Path

from tqdm import tqdm
import hydra
import torch
from torch.optim.lr_scheduler import StepLR

from model import build_model
from dataset import get_loaders, split_wsi, split_data
from loss import get_loss_fn
from radam import RAdam
from saver import Checkpoint


def train(
        model, device, train_loader, optimizer, scheduler, criterion, epoch,
        epochs, chunk, chunks, ckpt):
    model.train()
    desc = f"Ep [{epoch} / {epochs}] Chunk [{chunk} / {chunks}]"
    total = len(train_loader)
    with tqdm(enumerate(train_loader), desc=desc, total=total) as t:
        for batch_idx, data in t:
            data = data.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, data)
            loss.backward()
            optimizer.step()
            ckpt.log(
                "training loss", loss, epoch, chunk, chunks, batch_idx, total)
            t.set_postfix({
                "loss": f"{loss.item():6f}",
                "done": f"{batch_idx * len(data)}"
            })
            t.update(1)

    ckpt.save(model, optimizer, scheduler, epoch, chunk, loss)


@hydra.main(config_path="config", config_name="config")
def main(cfg):
    torch.cuda.empty_cache()
    torch.manual_seed(cfg.param.seed)

    # Training settings
    cwd = Path(hydra.utils.get_original_cwd())
    wsi_dir = cwd/cfg.dir.wsi
    patch_dir = cwd/cfg.dir.patch
    ckpt = Checkpoint(
        cwd, cfg.gpus, cfg.dir.resume, cfg.dir.save_to, cfg.log.save_model)

    device = torch.device(
        f"cuda:{cfg.gpus[0]}" if cfg.gpus[0] != -1 else "cpu")

    model = build_model(gpus=cfg.gpus)
    optimizer = RAdam(model.parameters(), lr=cfg.param.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=cfg.param.gamma)
    if cfg.dir.resume:
        model, optimizer, scheduler = ckpt.load_state(
            model, optimizer, scheduler)
    criterion = get_loss_fn()

    train_wsi, test_wsi = split_wsi(
        wsi_dir, ckpt.save_to, cwd, ratio=cfg.data.ratio,
        projects=cfg.data.projects, strategies=cfg.data.strategies,
        limit=cfg.data.limit)
    for epoch in range(ckpt.start_epoch, cfg.param.epochs + 1):
        split_data(
            patch_dir, ckpt.save_to, train_wsi, test_wsi, cfg.data.chunks,
            epoch, cfg.dir.resume)
        for chunk in range(ckpt.start_chunk, cfg.data.chunks):
            data_loader = get_loaders(
                cfg.param.batch_size, ckpt.save_to, chunk, cfg.gpus)
            train(
                model, device, data_loader, optimizer, scheduler, criterion,
                epoch, cfg.param.epochs, chunk, cfg.data.chunks, ckpt)

        ckpt.start_chunk = 0
        scheduler.step()
        ckpt.save(model, optimizer, scheduler, epoch, chunk, loss=False)

    ckpt.close_writer()


if __name__ == '__main__':
    main()
