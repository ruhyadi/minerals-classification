import pyrootutils

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator=[".git", "pyproject.toml"],
    pythonpath=True,
    dotenv=True,
)

import json
import os
from glob import glob
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from PIL import Image
from pytorch_lightning import LightningModule
from torchmetrics import ConfusionMatrix
from torchvision import transforms

from src import utils

log = utils.get_pylogger(__name__)

categories = {
    0: "biotite",
    1: "bornite",
    2: "chrysocolla",
    3: "malachite",
    4: "muscovite",
    5: "pyrite",
    6: "quartz",
}


@hydra.main(version_base="1.2", config_path=root / "configs", config_name="inference.yaml")
def inference(cfg: DictConfig):
    """Inference function."""

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    if Path(cfg.get("weights_path")).suffix == ".ckpt":
        model = model.load_from_checkpoint(cfg.weights_path)
    else:
        model.load_state_dict(torch.load(cfg.weights_path))
    model.load_from_checkpoint(cfg.get("weights_path"))
    model.eval().to(cfg.get("device"))

    # preprocesssing torch transforms
    preprocess = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    imgs_path = sorted(glob(os.path.join(cfg.get("source_dir"), "*.jpg")))
    for img_path in imgs_path:
        img = Image.open(img_path)
        img = torch.unsqueeze(preprocess(img), dim=0).to(
            cfg.get("device")
        )  # preprocess and add batch dim
        pred = model(img)
        pred = torch.nn.functional.softmax(pred, dim=1)[0]
        pred_idx = torch.argmax(pred).item()
        pred_conf = round(pred[pred_idx].item(), 2)
        pred_class = categories[pred_idx]
        print(f"Prediction: {pred_class} ({pred_conf})")

        # dumping to json
        with open(os.path.join(cfg.get("output_dir"), "predictions.json"), "w") as f:
            json.dump({img_path.split("/")[-1]: {"class": pred_class, "confidence": pred_conf}}, f)

    print("[INFO] Results saved to", cfg.get("output_dir"))


if __name__ == "__main__":
    inference()
