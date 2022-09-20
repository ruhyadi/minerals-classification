"""Minerals dataset datamodule"""

from typing import Any, Dict, Optional, Tuple
from glob import glob
from PIL import Image
import cv2

import torch
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms

class MineralsDataModule(LightningDataModule):
    """Minerals datamodule"""

    def __init__(
        self,
        data_dir: str = "data/minet",
        train_size: float = 0.8,
        batch_size: int = 64,
        num_workers: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # data transformation/augmentation
        self.transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None

    @property
    def num_classes(self) -> int:
        return 7

    def setup(self, stage: Optional[str] = None):
        """Load and split dataset from MineralsDataset"""
        self.dataset = MineralsDataset(self.hparams.data_dir, self.transforms)
        train_size = int(self.hparams.train_size * len(self.dataset))
        val_size = len(self.dataset) - train_size
        self.data_train, self.data_val = random_split(
            dataset=self.dataset, 
            lengths=[train_size, val_size]
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )

class MineralsDataset(Dataset):
    """Minerals dataset"""

    def __init__(
        self,
        data_dir: str = "data/minet",
        transform: Optional[transforms.Compose] = None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.transform = transform if transform else transforms.ToTensor()

        self.categories = {
            "biotite": 0, "bornite": 1, "chrysocolla": 2,
            "malachite": 3, "muscovite": 4, "pyrite": 5, 
            "quartz": 6
        }

        self.images_path, self.labels = self.load_data(data_dir)

    def __len__(self) -> int:
        return len(self.images_path)

    def __getitem__(self, idx: int) -> Any:
        """Get item at index idx"""
        image_path = self.images_path[idx]
        label = self.labels[idx]

        # load image
        image = Image.open(image_path)
        image = image.convert("RGB") if image.mode != "RGB" else image
        image = self.transform(image)

        # transforms labels to tensor
        label = torch.tensor(self.categories[label])

        return image, label

    def load_data(self, data_dir: str) -> Tuple[list, list]:
        """Get image path and label from data_dir"""
        extensions = ["jpg", "jpeg", "png"]

        images_path = []
        [images_path.extend(glob(f"{data_dir}/*/*.{ext}")) for ext in extensions]
        labels = [path.split("/")[-2] for path in images_path]

        return [images_path, labels]

if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "datamodule" / "minerals.yaml")
    cfg.data_dir = str(root / "data/minet")
    cfg.batch_size = 1
    cfg.num_workers = 0

    dataset = hydra.utils.instantiate(cfg)
    dataset.setup()
    dataloader = dataset.train_dataloader()

    print(len(dataloader))