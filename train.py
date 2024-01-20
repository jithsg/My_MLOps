from data import CIFAR10DataModule 
from model import LitModel
from dotenv import load_dotenv
import logging
import os
import wandb
import pytorch_lightning as pl
# your favorite machine learning tracking tool
from pytorch_lightning.loggers import WandbLogger
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import random_split, DataLoader

from torchmetrics import Accuracy

from torchvision import transforms
from torchvision.datasets import CIFAR10
load_dotenv()  # This loads the .env file at the script's start

wandb_api_key = os.getenv('WANDB_API_KEY')
wandb.login(key=wandb_api_key)
logger = logging.getLogger(__name__)


class ImagePredictionLogger(pl.callbacks.Callback):
    def __init__(self, val_samples, num_samples=32):
        super().__init__()
        self.num_samples = num_samples
        self.val_imgs, self.val_labels = val_samples
    
    def on_validation_epoch_end(self, trainer, pl_module):
        # Bring the tensors to CPU
        val_imgs = self.val_imgs.to(device=pl_module.device)
        val_labels = self.val_labels.to(device=pl_module.device)
        # Get model prediction
        logits = pl_module(val_imgs)
        preds = torch.argmax(logits, -1)
        # Log the images as wandb Image
        trainer.logger.experiment.log({
            "examples":[wandb.Image(x, caption=f"Pred:{pred}, Label:{y}") 
                           for x, pred, y in zip(val_imgs[:self.num_samples], 
                                                 preds[:self.num_samples], 
                                                 val_labels[:self.num_samples])]
            })
        






@hydra.main(config_path='./configs', config_name='config', version_base=None)
def main(cfg: DictConfig):
    # Data Module
    dm = CIFAR10DataModule(batch_size=cfg.datamodule.batch_size)
    dm.prepare_data()
    dm.setup()

    # Get validation samples
    val_samples = next(iter(dm.val_dataloader()))
    val_imgs, val_labels = val_samples[0], val_samples[1]
    val_imgs.shape, val_labels.shape

    # Model
    model = LitModel(cfg.model.input_shape, dm.num_classes)

    # Logger
    wandb_logger = WandbLogger(project=cfg.logger.project, entity=cfg.logger.entity)

    # Callbacks
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="train_loss", patience=3, verbose=True, mode="min"
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(dirpath="./models",
        filename="best-checkpoint",
        monitor="train_loss",
        mode="min",)

    # Trainer
    trainer = pl.Trainer(max_epochs=cfg.trainer.max_epochs,
                         logger=wandb_logger,
                         callbacks=[early_stop_callback,
                                    ImagePredictionLogger(val_samples),
                                    checkpoint_callback])

    # Train and Test
    trainer.fit(model, dm)
    trainer.test(dataloaders=dm.test_dataloader())

    # Close wandb run
    wandb.finish()

if __name__ == "__main__":
    main()
