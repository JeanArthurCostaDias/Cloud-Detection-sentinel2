from sklearn.model_selection import train_test_split, KFold
import segmentation_models_pytorch as smp
import torch
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from osgeo import gdal, osr
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from torch.utils.data import WeightedRandomSampler
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.tuner.tuning import Tuner
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torchvision.transforms as T
import pytorch_lightning as pl
import os
import cv2
import gc

torch.manual_seed(42)
np.random.seed(42)

dataset = "./38-Cloud_training"
data_dir = pathlib.Path(dataset)

train_blue = sorted(list(data_dir.glob("train_blue/*.TIF")))
train_green = sorted(list(data_dir.glob("train_green/*.TIF")))
train_red = sorted(list(data_dir.glob("train_red/*.TIF")))
train_nir = sorted(list(data_dir.glob("train_nir/*.TIF")))

train_images = sorted(list(zip(train_blue, train_green, train_red, train_nir)))
train_ground_truth = sorted(list(data_dir.glob("train_gt/*.TIF")))
train_images, val_images, train_masks, val_masks = train_test_split(train_images, train_ground_truth, test_size=0.15, random_state=42)


class CloudDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.images = image_paths
        self.masks = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        blue = gdal.Open(str(self.images[idx][0]))
        green = gdal.Open(str(self.images[idx][1]))
        red = gdal.Open(str(self.images[idx][2]))
        nir = gdal.Open(str(self.images[idx][3]))

        image = np.stack([blue.ReadAsArray(), green.ReadAsArray(), red.ReadAsArray(), nir.ReadAsArray()], axis=0)
        mask = gdal.Open(str(self.masks[idx])).ReadAsArray() / 255
        image = torch.tensor(np.array(image), dtype=torch.float32) / 65535 + 1e-6  # Normalização
        mask = torch.tensor(np.array(mask), dtype=torch.long).unsqueeze(0)  # Máscara como inteiro (1 canal)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(degrees=30)
])


from torchmetrics.classification import Dice, JaccardIndex
from torchmetrics import Accuracy, Precision, Recall
import torch.nn.functional as F
from sklearn.metrics import f1_score, roc_auc_score

class CloudModel(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, lr=1e-5, **kwargs):
        super(CloudModel, self).__init__()
        self.model = smp.create_model(
            arch,
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=out_classes,
            **kwargs,
        )
        self.arch = arch.lower()
        self.save_hyperparameters()
        
        self.criterion = nn.BCEWithLogitsLoss()
        self.dice = Dice(num_classes=2)
        self.iou = JaccardIndex(task="binary", average='weighted')
        self.accuracy = Accuracy(task="binary", average='weighted')
        self.precision = Precision(task="binary", average='weighted', num_classes=2)
        self.recall = Recall(task="binary", average='weighted', num_classes=2)

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
        
        loss = self.criterion(y_hat, y.float())

        self.training_step_outputs.append(loss)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    # ===================== OTIMIZADOR =======================

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
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

        loss = self.criterion(y_hat, y.float()).detach()
        self.validation_step_outputs.append(loss)
        self.log("val_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return loss

    def on_validation_epoch_end(self):
        epoch_average = torch.mean(torch.tensor(self.validation_step_outputs))
        self.log("validation_epoch_average", epoch_average)
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()

        # Métricas focadas no desbalanceamento
        dice_score = self.dice(preds, y)
        iou_score = self.iou(preds, y)
        recall_score = self.recall(preds, y)  # Recall importante para detectar a minoria
        precision_score = self.precision(preds, y)  
        f1_weighted = f1_score(y.cpu().numpy(), preds.cpu().numpy(), average='weighted')
        auc_roc = roc_auc_score(y.cpu().numpy(), probs.cpu().numpy())

        self.log("test_dice", dice_score, prog_bar=True, sync_dist=True)
        self.log("test_iou", iou_score, prog_bar=True, sync_dist=True)
        self.log("test_recall", recall_score, prog_bar=True, sync_dist=True)
        self.log("test_precision", precision_score, prog_bar=True, sync_dist=True)
        self.log("test_f1_weighted", f1_weighted, prog_bar=True, sync_dist=True)
        self.log("test_auc_roc", auc_roc, prog_bar=True, sync_dist=True)

        return {"dice": dice_score, "iou": iou_score, "recall": recall_score, "precision": precision_score, "f1": f1_score}

    def on_test_epoch_end(self):
        epoch_average = torch.mean(torch.tensor(self.test_step_outputs))
        self.log("test_epoch_average", epoch_average)
        self.test_step_outputs.clear()

num_folds = 3
kf = KFold(n_splits=num_folds, shuffle=True, random_state=42)
architectures =  ["unetplusplus","deeplabv3plus", "segformer"]

for arch in architectures:
    for fold, (train_index, val_index) in enumerate(kf.split(train_images)):
        model = CloudModel(arch, encoder_name="resnet34", in_channels=4, out_classes=1, lr=1e-5)
        train_images_fold = [train_images[i] for i in train_index]
        val_images_fold = [train_images[i] for i in val_index]
        train_masks_fold = [train_masks[i] for i in train_index]
        val_masks_fold = [train_masks[i] for i in val_index]

        train_dataset = CloudDataset(train_images_fold, train_masks_fold, transform)
        val_dataset = CloudDataset(val_images_fold, val_masks_fold, transform)

        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=5)
        val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=5)

        # Callbacks específicos do fold
        checkpoint_callback = ModelCheckpoint(
            monitor='validation_epoch_average',
            dirpath=f'checkpoints/{arch}_fold_{fold}',  # Adicione o nome da arquitetura
            filename="best_model",
            save_top_k=1,
            mode='min'
        )
        lr_scheduler_callback = LearningRateMonitor(logging_interval='step')

        early_stop_callback = EarlyStopping(
            monitor='validation_epoch_average',  # Monitorando a perda de validação
            patience=3,                          # Número de épocas sem melhoria para parar
            #verbose=True,                        # Exibe mensagens de log quando ocorrer o early stopping
            mode='min'                           # Estamos tentando minimizar a perda
        )
        logger = TensorBoardLogger("tb_logs", name=f"{model.arch}_fold_{fold}")

        # Configura o Trainer
        trainer = pl.Trainer(
            max_epochs=100,
            callbacks=[early_stop_callback, checkpoint_callback, lr_scheduler_callback],
            logger=logger,
            accelerator="gpu",
            log_every_n_steps=1,
            precision=16,
            strategy='ddp_find_unused_parameters_true',
            devices=1
        )
        trainer.fit(model, train_loader, val_loader)

        del model
        gc.collect()
        torch.cuda.empty_cache()
