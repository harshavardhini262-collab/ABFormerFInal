import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import csv
from tqdm import tqdm

from AB_Data import AB_Data
from model import PredictModel
from utils import score

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ================= CONFIG =================
class Config:
    DATA_FILE_PATH = "data/data.xlsx"
    EMBEDDING_PATHS = [
        "Embeddings/antibinder_heavy.pkl",
        "Embeddings/Light.pkl",
        "Embeddings/Antigen.pkl"
    ]
    

    RESULTS_CSV = "ABFormer_results.csv"

    NUM_LAYERS = 6
    D_MODEL = 512
    DFF = 512
    NUM_HEADS = 8
    VOCAB_SIZE = 18

    BATCH_SIZE = 8
    MAX_EPOCHS = 100
    PATIENCE = 30

    LEARNING_RATE = 1e-4
    SEEDS = [1]


# ================= SEED =================
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


# ================= LOSS =================
class BCEWithLogitsLossCustom(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets):
        targets = targets.unsqueeze(1)
        return F.binary_cross_entropy_with_logits(inputs, targets)


# ================= EVALUATE =================
def evaluate(model, loader, device, criterion):
    model.eval()

    all_outputs = []
    all_labels = []
    total_loss = 0

    with torch.no_grad():
        for x1, x1m, x2, x2m, t1, t2, t3, a1, a2, a3, t4, labels in loader:

            x1, x1m, x2, x2m = x1.to(device), x1m.to(device), x2.to(device), x2m.to(device)
            t1, t2, t3 = t1.to(device), t2.to(device), t3.to(device)
            a1, a2, a3 = a1.to(device), a2.to(device), a3.to(device)
            t4 = t4.to(device)
            labels = labels.to(device)

            outputs = model(x1, x1m, x2, x2m, t1, t2, t3, a1, a2, a3, t4)

            loss = criterion(outputs, labels.float())
            total_loss += loss.item()

            probs = torch.sigmoid(outputs)

            all_outputs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = score(all_labels, all_outputs)
    avg_loss = total_loss / len(loader)

    return avg_loss, metrics, all_labels, all_outputs


# ================= TRAIN =================
def train_one_seed(seed, config, device):

    set_seed(seed)

    dataset = AB_Data(config.DATA_FILE_PATH, config.EMBEDDING_PATHS)

    train_loader, val_loader, test_loader = dataset.get_dataloaders(
        seed=seed,
        batch_size=config.BATCH_SIZE
    )

    train_losses = []
    val_losses = []

    model = PredictModel(
        num_layers=config.NUM_LAYERS,
        d_model=config.D_MODEL,
        dff=config.DFF,
        num_heads=config.NUM_HEADS,
        vocab_size=config.VOCAB_SIZE
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # 🔥 Scheduler added
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=3,
        verbose=True
    )

    criterion = BCEWithLogitsLossCustom()

    best_auc = 0
    patience_counter = 0
    best_epoch = 0

    for epoch in tqdm(range(config.MAX_EPOCHS)):

        model.train()
        total_loss = 0

        for x1, x1m, x2, x2m, t1, t2, t3, a1, a2, a3, t4, labels in train_loader:

            x1, x1m, x2, x2m = x1.to(device), x1m.to(device), x2.to(device), x2m.to(device)
            t1, t2, t3 = t1.to(device), t2.to(device), t3.to(device)
            a1, a2, a3 = a1.to(device), a2.to(device), a3.to(device)
            t4 = t4.to(device)
            labels = labels.to(device)

            outputs = model(x1, x1m, x2, x2m, t1, t2, t3, a1, a2, a3, t4)

            loss = criterion(outputs, labels.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss_epoch = total_loss / len(train_loader)
        train_losses.append(train_loss_epoch)

        # ===== VALIDATION =====
        val_loss, val_metrics, _, _ = evaluate(model, val_loader, device, criterion)
        val_losses.append(val_loss)

        # 🔥 Scheduler step
        scheduler.step(val_metrics['roc_auc'])

        if val_metrics['roc_auc'] > best_auc:
            best_auc = val_metrics['roc_auc']
            best_epoch = epoch
            patience_counter = 0
            best_model = model.state_dict()
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1

        if patience_counter >= config.PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

    # ===== LOSS CURVE =====
    plt.figure()
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig("loss_curve.png")
    plt.close()

    # ===== LOAD BEST =====
    model.load_state_dict(best_model)

    # ===== TEST =====
    test_loss, test_metrics, test_labels, test_outputs = evaluate(model, test_loader, device, criterion)

    # ===== CONFUSION MATRIX =====
    preds = [1 if p > 0.5 else 0 for p in test_outputs]

    cm = confusion_matrix(test_labels, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()

    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    plt.close()

    results = {
        'seed': seed,
        'best_epoch': best_epoch,

        'val_loss': val_loss,
        'val_acc': val_metrics['accuracy'],
        'val_auc': val_metrics['roc_auc'],
        'val_prauc': val_metrics['pr_auc'],
        'val_sensitivity': val_metrics['sensitivity'],
        'val_specificity': val_metrics['specificity'],

        'test_loss': test_loss,
        'test_se': test_metrics['sensitivity'],
        'test_sp': test_metrics['specificity'],
        'test_mcc': test_metrics['mcc'],
        'test_acc': test_metrics['accuracy'],
        'test_auc': test_metrics['roc_auc'],
        'test_f1': test_metrics['f1_score'],
        'test_ba': test_metrics['balanced_accuracy'],
        'test_prauc': test_metrics['pr_auc'],
        'test_ppv': test_metrics['ppv'],
        'test_npv': test_metrics['npv'],
    }

    return results


# ================= MAIN =================
def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    config = Config()

    # 🔥 CSV HEADER
    with open(config.RESULTS_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'seed','best_epoch',
            'val_loss','val_acc','val_auc','val_prauc','val_sensitivity','val_specificity',
            'test_loss','test_se','test_sp','test_mcc','test_acc','test_auc',
            'test_f1','test_ba','test_prauc','test_ppv','test_npv'
        ])

    for seed in config.SEEDS:

        result = train_one_seed(seed, config, device)
        
        # 🔥 WRITE RESULTS
        with open(config.RESULTS_CSV, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(list(result.values()))

    print("✅ Training Completed")


if __name__ == "__main__":
    main()