import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from model import UNet3D
from data_loader import get_data_loader

class SBRTTrainer(pl.LightningModule):
    def __init__(self, lr=1e-4):
        super(SBRTTrainer, self).__init__()
        self.model = UNet3D(in_channels=1, out_channels=1)
        self.loss_fn = nn.MSELoss()  # For image synthesis, can change for segmentation
        self.lr = lr

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        preds = self(images)
        loss = self.loss_fn(preds, labels)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

if __name__ == "__main__":
    train_loader = get_data_loader("data", batch_size=2, mode="train")

    model = SBRTTrainer()
    trainer = pl.Trainer(max_epochs=50, accelerator="gpu", devices=1)
    trainer.fit(model, train_loader)
