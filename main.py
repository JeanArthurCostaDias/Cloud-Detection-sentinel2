from sklearn.model_selection import train_test_split
import segmentation_models_pytorch as smp
import torch
import numpy as np
import pathlib
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, LearningRateFinder
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl

from model import CloudModel
from dataset import CloudDataset
import torchvision.transforms as T
import gc


images_path = pathlib.Path("subscenes")
masks_path = pathlib.Path("masks")

images = sorted(list(images_path.glob("*.npy")))
masks = sorted(list(masks_path.glob("*.npy")))


def calcular_fração_nuvem(mask_path):
    mask = np.load(mask_path)  # Carrega a máscara
    mask_class = np.argmax(mask, axis=-1)  # Converte de one-hot para classes (H, W)
    cloud_fraction = np.mean(mask_class == 1)  # Proporção de pixels pertencentes à classe CLOUD
    del mask,mask_class
    gc.collect()
    return cloud_fraction


cloud_fractions = np.array([calcular_fração_nuvem(mask) for mask in masks])

# Cria rótulos para estratificação
num_bins = 5  
labels = np.digitize(cloud_fractions, bins=np.linspace(0, 1, num_bins))

# Agora podemos fazer a divisão estratificada
train_images, val_images, train_masks, val_masks, train_labels, val_labels = train_test_split(
    images, masks, labels, test_size=0.3, stratify=labels, random_state=1
)

val_images, test_images, val_masks, test_masks = train_test_split(
    val_images, val_masks, test_size=0.5, stratify=val_labels, random_state=1
)

transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(degrees=30)
])
train_dataset = CloudDataset(train_images, train_masks, transform)
val_dataset = CloudDataset(val_images, val_masks)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)

architectures =  ["unet","DeepLabV3Plus", "segformer"]

for arch in architectures:
    # Criação do modelo
    model = CloudModel(arch, encoder_name="efficientnet-b0", in_channels=6, out_classes=1, weights="imagenet", lr=1e-3)

    # Callbacks específicos do modelo
    checkpoint_callback = ModelCheckpoint(
        monitor='validation_epoch_average',
        dirpath=f'checkpoints/{arch}',  # Removido o fold do nome do diretório
        filename="best_model",
        save_top_k=1,
        mode='min'
    )
    lr_scheduler_callback = LearningRateMonitor(logging_interval='step')
    early_stop_callback = EarlyStopping(
        monitor='validation_epoch_average',  # Monitorando a perda de validação
        patience=10,                          # Número de épocas sem melhoria para parar
        mode='min'                           # Estamos tentando minimizar a perda
    )
    logger = TensorBoardLogger("tb_logs", name=f"{model.arch}")
    
    # Configura o Trainer
    trainer = pl.Trainer(
        max_epochs=100,
        callbacks=[early_stop_callback, checkpoint_callback, lr_scheduler_callback],
        logger=logger,
        accelerator="gpu",
        log_every_n_steps=1,
        precision="16-mixed",
        strategy='ddp_find_unused_parameters_true',
        devices=1
    )
    trainer.fit(model, train_loader, val_loader)

    # Limpeza após cada treinamento
    del model
    gc.collect()
    torch.cuda.empty_cache()

