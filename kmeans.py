import json
import random
from pathlib import Path
from collections import defaultdict

import hydra
import faiss
import pandas as pd
import torch
from torchvision.io import read_image
from torchvision.utils import save_image
import numpy as np
from tqdm import tqdm

from model import Encoder
from dataset import get_loaders
from saver import Coords2ASAP


def load_encoder(data_root, weight_path, device):
    encoder = Encoder()
    if weight_path:
        weight = torch.load(weight_path)
    else:
        weight = torch.load(get_best_weight(data_root))
    encoder.load_state_dict(weight)

    if device >= 0:
        encoder = encoder.to(f"cuda:{device}")
    encoder.eval()
    return encoder


def reduce_dim(
        chunks, encoder, saved_to, wsis, batch_size, device, dry_run=False):
    vecs, paths = [], []
    chunks = dry_run if dry_run else chunks
    with tqdm(range(chunks)) as t:
        for chunk in t:
            t.set_description(f"[Reducing Dim] [{chunk+1}/{chunks}]")
            vecs_, paths_ = reduce_dim_chunk(
                encoder=encoder, data_root=saved_to, wsis=wsis, chunk=chunk,
                batch_size=batch_size, device=device)
            vecs.extend(vecs_)
            paths.extend(paths_)
            t.update(1)
            if dry_run and chunk >= dry_run-1:
                break

    return np.array(vecs), np.array(paths)


def reduce_dim_chunk(
        encoder, data_root, wsis, chunk, batch_size=512, device=0):

    data_loader = get_loaders(
        batch_size, data_root, chunk=chunk, gpus=[device], is_train=False,
        with_name=True, wsis=wsis)

    vecs = []
    paths = []
    for imgs, paths_ in tqdm(data_loader, desc="Reduced"):
        # 3 * 224 * 224 -> 256 * 1 * 1
        encoded = encoder(imgs.to(f"cuda:{device}"))
        encoded = encoded.cpu().detach().numpy().reshape(len(imgs), -1)
        vecs.extend(encoded)
        paths.extend(paths_)

    return vecs, paths


def paths2coords(paths):
    return np.array([list(map(int, Path(i).stem.split("_"))) for i in paths])


def paths2slidenames(paths):
    return np.array([Path(i).resolve().parents[2].name for i in paths])


def pathsOfProj(paths, project_slides):
    paths_proj = []
    for i in paths:
        if Path(i).resolve().parents[2].name in project_slides:
            paths_proj.append(i)
    return paths_proj


def get_best_weight(data_root):
    result_dir = Path(data_root)
    with open(result_dir/"weight/best.txt", "r") as f:
        txt = f.read()
    ep = txt.split("@ep")[1].split()[0]
    chunk = txt.split("@chunk")[1].split()[0]
    return result_dir/"weight"/f"encoder{ep}_{chunk}.pth"


def run_kmeans(vecs, ncentroids=10, niter=20, device=0, verbose=True):
    dim = vecs.shape[1]
    if device == -1:
        print("  On CPU")
        kmeans = faiss.Kmeans(dim, ncentroids, niter=niter, verbose=verbose)
        kmeans.train(vecs)
        distances, groups = kmeans.index.search(vecs, 1)
    else:
        print("  On GPU")
        if vecs.sum() == 0:
            msg = "All Image has no value. "
            msg += "Please retry with the other weight."
            print(msg)
        clus = faiss.Clustering(dim, ncentroids)
        clus.verbose = verbose
        clus.niter = niter
        clus.max_points_per_centroid = 10000000

        res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16 = False
        cfg.device = device

        index = faiss.GpuIndexFlatL2(res, dim, cfg)

        clus.train(vecs, index)
        distances, groups = index.search(vecs, 1)

    return groups


def save_as_grid(
        paths, data_root, save_to, indices, projects, strategies,
        ncentroids, img_per_grid=50, nrow=5):
    save_to = Path(save_to)/"grids"
    if not save_to.exists():
        save_to.mkdir()
    indices = indices.ravel()
    cwd = Path(hydra.utils.get_original_cwd())

    slides = defaultdict(set)
    for proj in projects:
        for strategy in strategies:
            manifest = f"tcga_manifests/{proj}_{strategy}.txt"
            df = pd.read_csv(f"{cwd}/{manifest}", delimiter="\t")
            slides[proj] |= set([Path(i).stem for i in df.filename])

    for grp in tqdm(range(ncentroids)):
        indice = (indices == grp)
        if indice.sum() == 0:
            continue
        paths_grp = paths[indice]
        random.shuffle(paths_grp)
        for proj in projects:
            paths_proj = pathsOfProj(paths_grp, slides[proj])
            paths_grid = paths_proj[:img_per_grid]
            if len(paths_grid) == 0:
                continue
            imgs_grid = [read_image(i)/255 for i in paths_grid]
            save_image(
                imgs_grid, f"{save_to}/cluster_{grp}_{proj}.jpg", nrow=nrow,
                padding=1)


def save_ASAP(paths, data_root, save_to, grps, tile_size):
    save_to = Path(save_to)/"ASAP"
    if not save_to.exists():
        save_to.mkdir()
    slidenames = paths2slidenames(paths)
    coords = paths2coords(paths)
    with open(Path(data_root, "wsis.json"), "r") as f:
        wsis = json.load(f)
    slides = wsis["train"] + wsis["test"]
    for slide in tqdm(slides):
        indice = slidenames == slide
        if indice.sum() == 0:
            continue
        converter = Coords2ASAP()
        converter.register_annotations(
            coords[indice], grps[indice], tile_size=tile_size)
        converter.save_xml(f"{save_to}/{slide}.xml")


@hydra.main(config_path="config", config_name="config")
def main(cfg):
    assert cfg.dir.saved_to, "You need to set training result directory."

    encoder = load_encoder(
        cfg.dir.saved_to, cfg.dir.weight, cfg.gpus.autoencoder)
    vecs, paths = reduce_dim(
        cfg.data.chunks, encoder, cfg.dir.saved_to, cfg.param.wsis,
        cfg.param.batch_size, cfg.gpus.autoencoder, cfg.dry_run)

    print("Clustring with k-means")
    groups = run_kmeans(
        vecs, cfg.param.ncentroids, cfg.param.niter, cfg.gpus.kmeans)

    if cfg.save.grid:
        print("Making Grid Image")
        save_as_grid(
            paths, cfg.dir.saved_to, cfg.dir.save_to, groups,
            cfg.data.projects, cfg.data.strategies, cfg.param.ncentroids,
            cfg.param.img_per_grid, cfg.param.nrow)

    print("Saving Results")
    save_ASAP(
        paths, cfg.dir.saved_to, cfg.dir.save_to, groups,
        tile_size=cfg.patch.size)


if __name__ == "__main__":
    main()
