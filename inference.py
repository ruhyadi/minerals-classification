import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

from typing import List, Optional, Tuple
import os
from glob import glob
import numpy as np

import torch
from torchvision import transforms
import cv2
from PIL import Image

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import LightningLoggerBase


from src import utils

log = utils.get_pylogger(__name__)


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="inference.yaml")
def inference(cfg: DictConfig):
    """Inference function"""

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    # model.load_state_dict(torch.load(cfg.get("weights_path")))
    model.load_from_checkpoint(cfg.get("weights_path"))
    model.eval().to(cfg.get("device"))

    # preprocesssing torch transforms
    preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    
    imgs_path = sorted(glob(os.path.join(cfg.get("source_dir"), "*.jpg")))
    for img_path in imgs_path:
        img = Image.open(img_path)
        img = torch.unsqueeze(preprocess(img), dim=0).to(cfg.get("device")) # preprocess and add batch dim
        pred = model(img)
        print("Prediction 0:", pred)
        pred = torch.argmax(pred, dim=1)
        print("Prediction 1:", pred)

if __name__ == "__main__":
    inference()