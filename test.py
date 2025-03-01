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

torch.manual_seed(42)
np.random.seed(42)

images_path = pathlib.Path("subscenes")
masks_path = pathlib.Path("masks")

images = sorted(list(images_path.glob("*.npy")))
masks = sorted(list(masks_path.glob("*.npy")))

train_images, val_images, train_masks, val_masks = train_test_split(images, masks, test_size=0.3, random_state=42)
val_images, test_images, val_masks, test_masks = train_test_split(val_images, val_masks, test_size=0.5, random_state=42)

transform = T.Compose([
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(degrees=30)
])

test_dataset = CloudDataset(test_images,test_masks,transform)

test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
# Inicializa o defaultdict para armazenar os resultados
resultados = defaultdict(dict)

# Suponha que você tenha uma lista de arquiteturas
architectures = ['unet', 'DeepLabV3Plus', 'segformer']  # Exemplo de arquiteturas

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

# Converter os resultados para um formato adequado para o CSV
# Aqui estou assumindo que cada 'predictions' contém um dicionário ou lista de métricas
# Ajuste de acordo com a estrutura real do seu 'predictions'
df = pd.DataFrame.from_dict(resultados, orient='index')

# Salva o DataFrame como CSV
df.to_csv('resultados.csv')

