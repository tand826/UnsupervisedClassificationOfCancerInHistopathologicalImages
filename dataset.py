import json
import random
from pathlib import Path

from tqdm import tqdm
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image


class TCGADataset(Dataset):

    def __init__(self, csv_path, wsis=None, transform=None, with_name=False):
        csv = pd.read_csv(csv_path, names=["path"])
        if wsis:
            paths = []
            for wsi in wsis:
                paths.append(csv[csv["path"].str.startswith(wsi)])
            self.paths = pd.concat(paths, ignore_index=True)
        else:
            self.paths = csv
        assert len(self.paths) != 0, "No patches found."
        self.transform = transform
        self.with_name = with_name

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        tensor = read_image(self.paths.iloc[idx, 0])
        tensor = tensor.type(torch.float32).div_(255)
        if not self.with_name:
            return tensor
        else:
            return tensor, self.paths.iloc[idx, 0]


def split_wsi(
        root, save_to, cwd, ratio=0.9, projects=["*"],
        strategies=["diagnostic_slides"], limit=-1):
    train_wsi = []
    test_wsi = []
    for proj in projects:
        wsis = [i.stem for i in Path(root).glob(f"{proj}/*/*.svs")]
        for strategy in strategies:
            csv = pd.read_csv(
                f"{cwd}/tcga_manifests/{proj}_{strategy}.txt", delimiter="\t")
            strategy_slides = [Path(name).stem for name in csv.filename]
            wsis = list(set(wsis) & set(strategy_slides))
        wsis = wsis[:limit]
        random.shuffle(wsis)
        train_wsi.extend(wsis[:int(len(wsis)*ratio)])
        test_wsi.extend(wsis[int(len(wsis)*ratio):])
    with open(f"{save_to}/wsis.json", "w") as f:
        json.dump({"train": train_wsi, "test": test_wsi}, f)
    return train_wsi, test_wsi


def split_data(
        root, save_to, train_wsis, test_wsis, chunk_num, epoch, resume=False):
    if resume:
        print(f"Resuming from {resume}")
        return
    if epoch != 1:
        print("Resuming")
        return
    for phase, wsis in {"train": train_wsis, "test": test_wsis}.items():
        with tqdm(wsis, total=len(wsis)) as t:
            patch_in_chunk = 0
            for wsi in t:
                t.set_description(f"[Dataset] {wsi}")
                parents = list(Path(root).glob(f"*/{wsi}"))
                if not parents:
                    print(f"{wsi} is not in the patch directory")
                    continue
                parent = parents[0]
                paths = list(parent.glob("patches/foreground/*.jpg"))
                indices = list(range(len(paths)))
                random.shuffle(indices)

                csvs = {
                    i: open(f"{save_to.chunk}/{phase}/chunk_{i}.csv", "a")
                    for i in range(chunk_num)
                }
                chunk = 0
                for idx in indices:
                    csvs[chunk].write(f"{paths[idx]}\n")
                    chunk = (chunk + 1) % chunk_num
                for csv in csvs.values():
                    csv.close()

                patch_in_chunk += len(paths) / chunk_num
                t.set_postfix(patch_in_chunk=patch_in_chunk)
                t.update(1)


def get_loaders(
        batch_size, save_to, chunk, gpus, is_train=True, with_name=False,
        wsis=None):
    train_kwargs = {'batch_size': batch_size}
    if gpus[0] != -1:
        cuda_kwargs = {
            'num_workers': 2,
            'pin_memory': True,
            'shuffle': is_train}
        train_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    phase = "train" if is_train else "test"
    dataset = TCGADataset(
        f"{save_to}/chunk/{phase}/chunk_{chunk}.csv",
        wsis=wsis, transform=transform, with_name=with_name)
    return DataLoader(dataset, **train_kwargs)
