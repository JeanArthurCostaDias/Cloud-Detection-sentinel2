from sklearn.model_selection import train_test_split
import torch
import numpy as np
import pathlib
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from model import CloudModel
from dataset import CloudDataset
import torchvision.transforms as T
from collections import defaultdict
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
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
_, val_images, _, val_masks, _, val_labels = train_test_split(
    images, masks, labels, test_size=0.3, stratify=labels, random_state=1
)

_, test_images, _, test_masks = train_test_split(
    val_images, val_masks, test_size=0.5, stratify=val_labels, random_state=1
)

test_dataset = CloudDataset(test_images,test_masks)

test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)

resultados = defaultdict(dict)

architectures = ['unet', 'DeepLabV3Plus', 'segformer']

cloud_fractions_test = np.array([calcular_fração_nuvem(mask) for mask in test_masks])

lower_bound = 0.2
upper_bound = 0.5

intermediate_cloud_indices = np.where((cloud_fractions_test >= lower_bound) & (cloud_fractions_test <= upper_bound))[0]

num_samples = 5
random_indices = np.random.choice(intermediate_cloud_indices, num_samples, replace=False)

# Seleciona as imagens e máscaras correspondentes
sample_images = [test_images[i] for i in random_indices]
sample_masks = [test_masks[i] for i in random_indices]

print(sample_images)
print("==========")
print(sample_masks)

# Cria um dataset apenas para essas imagens
sample_dataset = CloudDataset(sample_images, sample_masks, transform=None)
sample_loader = DataLoader(sample_dataset, batch_size=1, shuffle=False)


# Percorre as arquiteturas e coleta os resultados
for arch in architectures:
    model_path = f'checkpoints/{arch}/best_model.ckpt'
    model = CloudModel.load_from_checkpoint(model_path, map_location=torch.device('cuda'))
    model.eval()
    
    # Inicializa o Trainer e testa
    trainer = pl.Trainer(
        max_epochs=100,
        accelerator="gpu",
        log_every_n_steps=1,
        precision=16,
        strategy='ddp_find_unused_parameters_true',
        devices=1
    )
    predictions = trainer.test(model=model,dataloaders=test_loader)

    
    # Armazena os resultados para cada arquitetura
    resultados[arch] = predictions

    for idx, (image, mask) in enumerate(sample_loader):
        with torch.no_grad():
            prediction = model(image)  # Faz a previsão

        #probs = F.softmax(prediction, dim=1)
        #preds = torch.argmax(probs, dim=1)

        probs = torch.sigmoid(prediction)
        preds = probs > 0.5

        image_np = image.squeeze().cpu().permute(1, 2, 0).numpy()[...,[0,1,2]]
        mask_np = mask.squeeze().cpu()

        pred_np = preds.squeeze((0,1)).cpu()

        # Plota e salva a imagem, máscara real e predição
        fig, axs = plt.subplots(1, 3, figsize=(12, 4))
        axs[0].imshow(image_np)
        axs[0].set_title("Imagem")
        axs[1].imshow(mask_np, cmap='gray')
        axs[1].set_title("Máscara Real")
        axs[2].imshow(pred_np, cmap='gray')
        axs[2].set_title(f"Predição - {arch}")

        for ax in axs:
            ax.axis('off')

        plt.savefig(f'predictions/{arch}_sample_{idx}.png')
        plt.close()



df = pd.DataFrame.from_dict(resultados, orient='index')

# Salva o DataFrame como CSV
df.to_csv('resultados.csv')

