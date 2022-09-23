"""Convert checkpoint to model."""
import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import os
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningModule

from src import utils

log = utils.get_pylogger(__name__)


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="convert.yaml")
def convert(cfg: DictConfig):
    """Convert checkpoint weights to model."""
    assert Path(cfg.get("weights_path")).suffix == ".ckpt", "Weights path must be a .ckpt file"
    assert cfg.get("convert_to") in ["pytorch", "onnx"], "Please Choose one of [pytorch, onnx]"

    ckpt_path = cfg.get("weights_path")
    if ckpt_path and not os.path.isabs(ckpt_path):
        ckpt_path = cfg.get(os.path.join(hydra.utils.get_original_cwd(), ckpt_path))
    if not os.path.exists(cfg.get("save_path")):
        os.makedirs(cfg.get("save_path"))

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    model = model.load_from_checkpoint(ckpt_path)

    if cfg.get("convert_to") == "pytorch":
        torch.save(model.state_dict(), f"{cfg.get('save_path')}.pt")
        log.info(f"Saved model weights to {cfg.get('save_path')}")
    elif cfg.get("convert_to") == "onnx":
        dummy_input = torch.randn(1, 3, 224, 224, device="cuda")
        model.cuda()
        torch.onnx.export(model, dummy_input, f"{cfg.get('save_path')}.onnx", verbose=True)
        log.info(f"Saved model weights to {cfg.get('save_path')}")


if __name__ == "__main__":
    convert()
