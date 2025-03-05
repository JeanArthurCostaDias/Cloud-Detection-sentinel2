import os
import numpy as np
import rasterio
import cv2
from sklearn.metrics import jaccard_score, precision_score, recall_score, accuracy_score

# 🚀 Configurações
GT_FOLDER_PATH = "38-Cloud_test/Entire_scene_gts"
PREDS_FOLDER_PATH = "preds_folder_root/preds_folder_deeplabv3plus"
PATCH_SIZE = (384, 384)
THRESH = 12 / 255  # Mesma binarização do MATLAB

# 🔍 Função para extrair IDs únicos das cenas
def extract_unique_sceneids(preds_folder):
    scene_ids = set()
    for filename in os.listdir(preds_folder):
        if filename.endswith(".TIF"):
            scene_id = "_".join(filename.split("_")[5:])  # Pegamos o ID da cena
            scene_ids.add(scene_id)
    return list(scene_ids)

# 🔍 Função para extrair posição do patch no nome do arquivo
def extract_rowcol_each_patch(filename):
    parts = filename.replace(".TIF", "").split("_")
    row, col = int(parts[1]), int(parts[2])  # Supondo padrão "patch_ROW_COL"
    return row, col

# 🔍 Função para reconstruir uma imagem completa a partir dos patches
def reconstruct_scene(scene_id, preds_folder):
    patches = {}
    max_row, max_col = 0, 0

    for filename in os.listdir(preds_folder):
        if scene_id in filename:
            row, col = extract_rowcol_each_patch(filename)
            max_row = max(max_row, row)
            max_col = max(max_col, col)

            patch_path = os.path.join(preds_folder, filename)
            with rasterio.open(patch_path) as src:
                patch = src.read(1)  # Lendo 1º canal
                patch = cv2.resize(patch, PATCH_SIZE)  # Garantindo tamanho fixo
                patch = (patch > THRESH).astype(np.uint8)  # Binarizando
                patches[(row, col)] = patch

    # Criar imagem grande com base no tamanho máximo dos patches
    H, W = (max_row + 1) * PATCH_SIZE[0], (max_col + 1) * PATCH_SIZE[1]
    reconstructed = np.zeros((H, W), dtype=np.uint8)

    for (row, col), patch in patches.items():
        y_start, x_start = row * PATCH_SIZE[0], col * PATCH_SIZE[1]
        reconstructed[y_start:y_start + PATCH_SIZE[0], x_start:x_start + PATCH_SIZE[1]] = patch

    return reconstructed

# 🔍 Função para calcular métricas
def compute_metrics(pred, gt):
    gt_flat, pred_flat = gt.flatten(), pred.flatten()
    metrics = {
        "Jaccard": jaccard_score(gt_flat, pred_flat, average='binary'),
        "Precision": precision_score(gt_flat, pred_flat, zero_division=0),
        "Recall": recall_score(gt_flat, pred_flat, zero_division=0),
        "Accuracy": accuracy_score(gt_flat, pred_flat),
    }
    return metrics

# 🚀 Executando para todas as cenas
scene_ids = extract_unique_sceneids(PREDS_FOLDER_PATH)
results = []

for scene_id in scene_ids:
    print(f"Processando cena {scene_id}...")

    # 🏷️ Ground Truth
    gt_path = os.path.join(GT_FOLDER_PATH, f"edited_corrected_gts_{scene_id}")
    with rasterio.open(gt_path) as src:
        gt_image = src.read(1)

    # 🏷️ Reconstrução
    pred_image = reconstruct_scene(scene_id, PREDS_FOLDER_PATH)

    # 🔍 Ajuste do tamanho da predição para coincidir com GT
    pred_image = pred_image[:gt_image.shape[0], :gt_image.shape[1]]

    # 📊 Calculando métricas
    metrics = compute_metrics(pred_image, gt_image)
    results.append([scene_id, metrics["Jaccard"], metrics["Precision"], metrics["Recall"], metrics["Accuracy"]])

    # 💾 Salvando máscara reconstruída
    save_path = os.path.join(PREDS_FOLDER_PATH, f"entire_mask_{scene_id}.TIF")
    with rasterio.open(save_path, 'w', driver='GTiff', height=pred_image.shape[0],
                       width=pred_image.shape[1], count=1, dtype=np.uint8) as dst:
        dst.write(pred_image, 1)

# 📊 Imprimindo média das métricas
mean_metrics = np.mean([list(m) for _, *m in results], axis=0)
print(f"Média das métricas: Jaccard={mean_metrics[0]:.3f}, Precisão={mean_metrics[1]:.3f}, Recall={mean_metrics[2]:.3f}, Acurácia={mean_metrics[3]:.3f}")