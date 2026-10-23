"""Locate OWSM model files from a Hugging Face repo id or a local ESPnet model directory."""

import os
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ModelFiles:
    name: str  # repo id or directory name; used to key the engine cache
    root: Path
    config: Path
    checkpoint: Path


def cache_dir() -> Path:
    """Where built TensorRT engines are stored (override with ``OWSM_TRT_CACHE``)."""
    default = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "owsm_trt"
    return Path(os.environ.get("OWSM_TRT_CACHE", default))


def _from_dir(root: Path, name: str) -> ModelFiles:
    meta = root / "meta.yaml"
    if meta.exists():
        m = yaml.safe_load(meta.read_text())
        config = root / m["yaml_files"]["s2t_train_config"]
        ckpt = root / m["files"]["s2t_model_file"]
    else:
        configs = sorted(root.glob("exp/*/config.yaml")) or sorted(root.glob("**/config.yaml"))
        if not configs:
            raise FileNotFoundError(f"no config.yaml under {root}")
        config = configs[0]
        ckpts = sorted(config.parent.glob("*.pth"))
        if not ckpts:
            raise FileNotFoundError(f"no .pth checkpoint next to {config}")
        ckpt = ckpts[0]
    return ModelFiles(name=name, root=root, config=config, checkpoint=ckpt)


def resolve(model: str | os.PathLike) -> ModelFiles:
    """``model`` is a local directory or a Hugging Face repo id such as ``espnet/owsm_v4_medium_1B``.

    From a repo only the files ``meta.yaml`` and the training config point to are downloaded (some
    repos hold several checkpoints).
    """
    path = Path(model).expanduser()
    if path.is_dir():
        return _from_dir(path.resolve(), path.resolve().name)
    from huggingface_hub import hf_hub_download, snapshot_download

    from .config import load_espnet_config

    repo = str(model)
    meta = yaml.safe_load(Path(hf_hub_download(repo, "meta.yaml")).read_text())
    config = meta["yaml_files"]["s2t_train_config"]
    raw = load_espnet_config(hf_hub_download(repo, config))
    files = ["meta.yaml", config, meta["files"]["s2t_model_file"], raw["bpemodel"]]
    stats = (raw.get("normalize_conf") or {}).get("stats_file")
    root = Path(snapshot_download(repo, allow_patterns=files + ([stats] if stats else [])))
    return _from_dir(root, repo)


def resolve_relative(files: ModelFiles, rel: str) -> Path:
    """Paths inside an ESPnet config (``bpemodel``, ``stats_file``) are relative to the repo root."""
    p = Path(rel)
    return p if p.is_absolute() else files.root / p
