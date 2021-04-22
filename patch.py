from pathlib import Path
import subprocess

from tqdm import tqdm
from wsiprocess.cli import main as wp
import hydra


def download_wsi(cfg):
    cwd = Path(hydra.utils.get_original_cwd())
    wsi_dir = Path(cfg.dir.wsi)
    if not wsi_dir.exists():
        wsi_dir.mkdir(parents=True)
    for project in cfg.data.projects:
        for strategy in cfg.data.strategies:
            save_to = cwd/wsi_dir/project
            if not save_to.exists():
                save_to.mkdir()
            manifest = f"{project}_{strategy}.txt"
            proc = subprocess.run([
                "./gdc-client",
                "download",
                "-m", cwd/"tcga_manifests"/manifest,
                "-d", str(save_to)
            ], cwd=cwd)


def to_patch(cfg):
    for project in cfg.data.projects:
        root = Path(cfg.dir.wsi)/project
        wsis = list(root.glob("**/*.svs"))
        for wsi in tqdm(wsis, desc=f"[{project}]"):
            if (Path(cfg.dir.patch)/wsi.stem/"results.json").exists():
                print(f"{wsi} already patched")
                continue
            try:
                wp([
                    "evaluation",
                    str(wsi),
                    "-pw", str(cfg.patch.size),
                    "-ph", str(cfg.patch.size),
                    "-of", str(cfg.patch.on_foreground),
                    "-st", str(Path(cfg.dir.patch)/project)
                ])
            except Exception as e:
                print(f"Something went wrong with {wsi}")
                with open("error.txt", "a") as f:
                    f.write(f"\n [{wsi}] {e}")


@hydra.main(config_path="config", config_name="config")
def main(cfg):
    print("Start Downloading WSIs from TCGA-Portal")
    download_wsi(cfg)
    print("Start Patching WSIs")
    to_patch(cfg)


if __name__ == "__main__":
    main()
