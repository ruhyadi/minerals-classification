"""Minerals Model Module."""

from typing import Any, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import wandb
from pytorch_lightning import LightningModule
from torchmetrics import ConfusionMatrix, MaxMetric, MeanMetric, PrecisionRecallCurve, ROC
from torchmetrics.classification.accuracy import Accuracy

categories = [
    "biotite",
    "bornite",
    "chrysocolla",
    "malachite",
    "muscovite",
    "pyrite",
    "quartz",
]


class MineralsLitModule(LightningModule):
    """Minerals model module."""

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        num_classes: int = 7,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        # model
        self.net = net
        self.net.fc = torch.nn.Linear(self.net.fc.in_features, self.hparams.num_classes)

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metrics
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        return loss, preds, y, logits

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, _ = self.step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, logits = self.step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets, "logits": logits}

    def validation_epoch_end(self, outputs: List[Any]):
        """Log validation metrics at end of validation epoch."""
        preds = torch.cat([x["preds"] for x in outputs])
        targets = torch.cat([x["targets"] for x in outputs])
        logits = torch.cat([x["logits"] for x in outputs])

        # wandb logging
        self.wandb_logger = self.logger.experiment

        # log best accuracy
        acc = self.val_acc.compute()
        self.val_acc_best(acc)
        self.log("val/acc_best", self.val_acc_best, on_step=False, on_epoch=True, prog_bar=True)

        # log metrics
        self._confusion_matrix(preds, targets, self.hparams.num_classes, self.device)
        self._precision_recall_curve(logits, targets, self.hparams.num_classes, self.device)
        self._roc_curve(logits, targets, self.hparams.num_classes, self.device)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets, logits = self.step(batch)

        # plot metrics
        self._confusion_matrix(preds, targets, self.hparams.num_classes, self.device, log=False)
        self._precision_recall_curve(logits, targets, self.hparams.num_classes, self.device, log=False)
        self._roc_curve(logits, targets, self.hparams.num_classes, self.device, log=False)

        return {"loss": loss, "preds": preds, "targets": targets, "logits": logits}

    def configure_optimizers(self):
        """Configure optimizers and schedulers."""
        optimizer = self.hparams.optimizer(params=self.parameters())
        scheduler = self.hparams.scheduler(optimizer=optimizer)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def _confusion_matrix(self, preds, targets, num_classes, device, log=True):
        """Compute confusion matrix."""
        confmat = ConfusionMatrix(num_classes=num_classes, normalize="true").to(device)
        confmat(preds, targets)
        confmat_df = pd.DataFrame(
            confmat.compute().cpu().numpy(), 
            columns=categories, 
            index=categories
        )
        confmat_img = sns.heatmap(confmat_df, annot=True, fmt=".2f").get_figure()
        if log:
            self.wandb_logger.log({"val/confmat": wandb.Image(confmat_img)})
        else:
            plt.savefig("confmat.png")
        plt.clf()  # reset confusion matrix chart

    def _precision_recall_curve(self, logits, targets, num_classes, device, log=True):
        """Compute precision recall curve."""
        pr_curve = PrecisionRecallCurve(num_classes=num_classes).to(device)
        pr_curve(logits, targets)
        precision, recall, _ = pr_curve.compute()
        for i in range(num_classes):
            plt.plot(recall[i].cpu().numpy(), precision[i].cpu().numpy(), label=categories[i])
        plt.legend()
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision Recall Curve")
        if log:
            self.wandb_logger.log({"val/pr_curve": wandb.Image(plt)})
        else:
            plt.savefig("pr_curve.png")
        plt.clf() # reset precision recall curve chart

    def _roc_curve(self, logits, targets, num_classes, device, log=True):
        """Compute roc curve."""
        roc_curve = ROC(num_classes=num_classes).to(device)
        roc_curve(logits, targets)
        fpr, tpr, _ = roc_curve.compute()
        for i in range(num_classes):
            plt.plot(fpr[i].cpu().numpy(), tpr[i].cpu().numpy(), label=categories[i])
        plt.legend()
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        if log:
            self.wandb_logger.log({"val/roc_curve": wandb.Image(plt)})
        else:
            plt.savefig("roc_curve.png")
        plt.clf() # reset roc curve chart


if __name__ == "__main__":
    import hydra
    import omegaconf
    import pyrootutils
    from torchsummary import summary

    root = pyrootutils.setup_root(__file__, pythonpath=True)
    cfg = omegaconf.OmegaConf.load(root / "configs" / "model" / "minerals.yaml")
    model = hydra.utils.instantiate(cfg)

    summary(model, (3, 224, 224), device="cpu")
