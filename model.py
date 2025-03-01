from torchmetrics.classification import JaccardIndex
from torchmetrics.segmentation import DiceScore
from torchmetrics import Accuracy, Precision, Recall
import torch.nn.functional as F
from sklearn.metrics import f1_score
import segmentation_models_pytorch as smp
import torch
import pytorch_lightning as pl

class CloudModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, lr=1e-4, weights="imagenet",**kwargs):
        super(CloudModel, self).__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            encoder_weights=weights,
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )
        self.arch = arch.lower()
        self.save_hyperparameters()
        
        self.criterion = smp.losses.TverskyLoss("multiclass",alpha=0.3, beta=0.7,gamma=2)
        self.dice = DiceScore(num_classes=3, average="weighted",input_format='index')
        self.iou = JaccardIndex(task="multiclass", num_classes=3, average="weighted")
        self.accuracy = Accuracy(task="multiclass", num_classes=3, average="weighted")
        self.precision = Precision(task="multiclass", num_classes=3, average="weighted")
        self.recall = Recall(task="multiclass", num_classes=3, average="weighted")


        # initialize step metics
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.lr = lr

    def forward(self, x):
        return self.model(x)
    
    # ===================== TREINAMENTO =======================

    def training_step(self, batch, batch_idx):
        """ Passo de treinamento """
        x, y = batch
        y_hat = self(x)
        y = torch.argmax(y, dim=1)  # Converte de one-hot para rótulos (agora y tem shape: (batch_size, height, width))

        loss = self.criterion(y_hat.float(), y.long())

        self.training_step_outputs.append(loss)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # ===================== OTIMIZADOR =======================

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.25, patience=5, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


    def on_train_epoch_end(self):
        epoch_average = torch.mean(torch.tensor(self.training_step_outputs))

        # Log da média de perda de treinamento no final da época
        self.log("training_epoch_average", epoch_average)

        self.training_step_outputs.clear()  # Free memory

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        y = torch.argmax(y, dim=1)  # Converte de one-hot para rótulos (agora y tem shape: (batch_size, height, width))
        loss = self.criterion(y_hat.float(), y.long()).detach()
        self.validation_step_outputs.append(loss)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        epoch_average = torch.mean(torch.tensor(self.validation_step_outputs))
        self.log("validation_epoch_average", epoch_average)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = torch.argmax(y, dim=1)  # Converte de one-hot para rótulos (agora y tem shape: (batch_size, height, width))
        logits = self(x)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        # Métricas focadas no desbalanceamento
        dice_score = self.dice(preds, y)
        iou_score = self.iou(preds, y)
        recall_score = self.recall(preds, y)  # Recall importante para detectar a minoria
        precision_score = self.precision(preds, y)
        accuracy_score = self.accuracy(preds,y)  

        self.log("test_dice", dice_score, prog_bar=True, sync_dist=True)
        self.log("test_iou", iou_score, prog_bar=True, sync_dist=True)
        self.log("test_recall", recall_score, prog_bar=True, sync_dist=True)
        self.log("test_precision", precision_score, prog_bar=True, sync_dist=True)
        self.log("test_accuracy", precision_score, prog_bar=True, sync_dist=True)

        return {"dice": dice_score, "iou": iou_score, "recall": recall_score, "precision": precision_score,"accuracy":accuracy_score}

    def on_test_epoch_end(self):
        epoch_average = torch.mean(torch.tensor(self.test_step_outputs))
        self.log("test_epoch_average", epoch_average)
        self.test_step_outputs.clear()