from torch.utils.data import Dataset
import numpy as np
import torch

def add_padding_to_multiple_of_1024(input_array):
    """
    Adiciona padding a um array numpy para garantir que suas dimensões de altura e largura
    sejam múltiplos de 1024.

    Parâmetro:
    - input_array (numpy.ndarray): Array de entrada com formato (C, H, W)

    Retorna:
    - numpy.ndarray: Array com padding adicionado, se necessário.
    - int: Padding aplicado na altura (H)
    - int: Padding aplicado na largura (W)
    """
    
    C, height, width = input_array.shape

    pad_height = (1024 - height % 1024) if height % 1024 != 0 else 0
    pad_width = (1024 - width % 1024) if width % 1024 != 0 else 0

    padded_array = np.pad(
        input_array,
        (
            (0, 0),  # Sem padding para o canal (C)
            (0, pad_height),  # Padding para a altura (H)
            (0, pad_width),  # Padding para a largura (W)
        ),
        mode="constant",
        constant_values=0,
    )

    return padded_array


class CloudDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.images = image_paths
        self.masks = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = np.load(self.images[idx])[...,[4,3,2,1]].transpose(2,0,1)
        image = torch.tensor(add_padding_to_multiple_of_1024(image),dtype=torch.float32)

        mask = np.load(self.masks[idx]).transpose(2,0,1)
        mask = torch.tensor(add_padding_to_multiple_of_1024(mask),dtype=torch.long)

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask
