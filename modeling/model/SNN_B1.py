import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import argparse
import warnings
from torch.utils.data import DataLoader
import pandas as pd
from .SiameseDataset import SiameseDataset
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np
from torchvision.models import resnet50
warnings.filterwarnings("ignore")

class ResNetBackbone(nn.Module):
    def __init__(self, embedding_dim=512, pretrained=True):
        super().__init__()
        # Load a ResNet‑50, optionally with ImageNet weights
        self.resnet = resnet50(pretrained=pretrained)
        # Replace the final FC so it outputs `embedding_dim` features instead of 1000 classes
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(in_features, embedding_dim)

    def forward(self, x):
        # Forward through ResNet‑50 up to the final FC
        x = self.resnet(x)               # → [B, embedding_dim]
        # L2‑normalize so embeddings lie on the hypersphere
        return F.normalize(x, p=2, dim=1)
    

class SiameseRelation(nn.Module):
    def __init__(self, backbone, embedding_dim, hidden_dim=256):
        super().__init__()
        self.backbone = backbone
        # input: [|e1–e2|, e1*e2] → 2D features per example
        self.net = nn.Sequential(
            nn.Linear(2*embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, img1, img2):
        e1 = self.backbone(img1)
        e2 = self.backbone(img2)
        # combine
        diff = torch.abs(e1 - e2)
        prod =    e1 * e2
        x    = torch.cat([diff, prod], dim=1)  # [B, 2D]
        score = self.net(x).squeeze(1)         # [B]
        return e1, e2, score


def train_siamese_network(train_dataloader,
                          val_dataloader,
                          num_epochs=20,
                          learning_rate=0.001,
                          num_train_batches=100,
                          num_val_batches=20,
                          early_stopping_patience=3,
                          classification_threshold=0.5,
                          embedding_dim=256,
                          run_dir="runs/siamese_B1.1_net",
                          checkpoint_dir="checkpoints"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("You are now training on", device)

    writer = SummaryWriter(run_dir)
    
    # Instantiate backbone and relation network
    backbone = ResNetBackbone(embedding_dim=embedding_dim, pretrained=True).to(device)
    model    = SiameseRelation(backbone, embedding_dim).to(device)

    # **Use BCEWithLogits for a single‐logit binary classifier**
    criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=4, verbose=True)

    best_val_loss = float('inf')
    epochs_no_improve = 0
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, num_epochs+1):
        # ---------- Training ----------
        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{num_epochs} Train")
        for batch_idx, (img1, img2, labels) in enumerate(train_bar):
            if batch_idx >= num_train_batches:
                break

            img1, img2 = img1.to(device), img2.to(device)
            # Convert labels to float for BCE
            labels = labels.to(device).float()

            optimizer.zero_grad()
            _, _, logits = model(img1, img2)            # [B] raw scores
            loss = criterion(logits, labels)             # BCEWithLogits
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / min(len(train_dataloader), num_train_batches)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)

        # ---------- Validation ----------
        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            val_bar = tqdm(val_dataloader, desc=f"Epoch {epoch}/{num_epochs} Val  ")
            for batch_idx, (img1, img2, labels) in enumerate(val_bar):
                if batch_idx >= num_val_batches:
                    break

                img1, img2 = img1.to(device), img2.to(device)
                labels = labels.to(device).float()

                _, _, logits = model(img1, img2)
                loss = criterion(logits, labels)
                val_loss += loss.item()

                # Convert logits to probabilities, then to binary preds
                probs = torch.sigmoid(logits)
                preds = (probs > classification_threshold).long()

                all_preds.extend(preds.cpu().tolist())
                # Cast float labels back to int for metrics
                all_labels.extend(labels.cpu().long().tolist())

                val_bar.set_postfix(loss=loss.item())

        avg_val_loss = val_loss / min(len(val_dataloader), num_val_batches)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        print(f"\nEpoch {epoch} — Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # ---------- Metrics ----------
        cm   = confusion_matrix(all_labels, all_preds)
        acc  = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec  = recall_score(all_labels, all_preds, zero_division=0)
        f1   = f1_score(all_labels, all_preds, zero_division=0)

        writer.add_scalar("Metrics/Accuracy",  acc,  epoch)
        writer.add_scalar("Metrics/Precision", prec, epoch)
        writer.add_scalar("Metrics/Recall",    rec,  epoch)
        writer.add_scalar("Metrics/F1",        f1,   epoch)
        print(f"Val Metrics — Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

        # ---------- Confusion Matrix Plot ----------
        plt.figure(figsize=(6,5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix — Epoch {epoch}")
        plt.xlabel("Pred"); plt.ylabel("True")
        cm_path = os.path.join(checkpoint_dir, f"cm_epoch_{epoch}.png")
        plt.savefig(cm_path); plt.close()
        cm_img = torch.tensor(plt.imread(cm_path)).permute(2,0,1)
        writer.add_image("Confusion_Matrix", cm_img, epoch)

        # ---------- ROC Curve Plot ----------
        fpr, tpr, _  = roc_curve(all_labels, probs.cpu().numpy(), pos_label=1)
        roc_auc      = roc_auc_score(all_labels, probs.cpu().numpy())
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0,1],[0,1],'r--')
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")
        roc_path = os.path.join(checkpoint_dir, f"roc_epoch_{epoch}.png")
        plt.savefig(roc_path); plt.close()
        writer.add_figure("ROC_Curve", plt.gcf(), epoch)

        # ---------- Scheduler & Early Stopping ----------
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            ckpt = os.path.join(checkpoint_dir, f"best_epoch_{epoch}.pt")
            torch.save(model.state_dict(), ckpt)
            print("Model improved; checkpoint saved.")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping.")
                break

    writer.close()


if __name__ == '__main__':

    # M7dsh y5af da bs 3l4an lw 3aizen t3mlo run mn el terminal xD
    # Argument Parser for Terminal Execution
    parser = argparse.ArgumentParser(description="Train a Siamese network for face verification for our Attendace System.")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of image pairs per batch.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader.")
    parser.add_argument("--num_epochs", type=int, default=20, help="Maximum number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument("--num_train_batches", type=int, default=100, help="Training iterations per epoch.")
    parser.add_argument("--num_val_batches", type=int, default=20, help="Validation iterations per epoch.")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Epochs to wait before early stopping.")
    parser.add_argument("--classification_threshold", type=float, default=0.5, help="Threshold on cosine similarity for classification.")
    # parser.add_argument("--margin", type=float, default=0.47, help="Margin for MarginCosineLoss.")
    parser.add_argument("--run_dir", type=str, default="/home/seif_elkerdany/projects/modeling/model/runs/siamese_B1.1_net", help="TensorBoard log directory.")
    parser.add_argument("--checkpoint_dir", type=str, default="/home/seif_elkerdany/projects/modeling/model/checkpoints/B1.1", help="Directory to save model checkpoints and plots.")
    
    args = parser.parse_args()

    train_df = pd.read_csv("/home/seif_elkerdany/projects/data/train_dataset_2.csv")
    val_df = pd.read_csv("/home/seif_elkerdany/projects/data/val_split_2.csv")

    train_dataset = SiameseDataset(train_df, train=True)
    val_dataset = SiameseDataset(val_df, train= False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    train_siamese_network(train_dataloader=train_dataloader,
                          val_dataloader=val_dataloader,
                          num_epochs=args.num_epochs,
                          learning_rate=args.learning_rate,
                          num_train_batches=len(train_df),
                          num_val_batches=len(val_df),
                          early_stopping_patience=args.early_stopping_patience,
                          classification_threshold=args.classification_threshold,
                          run_dir=args.run_dir,
                          checkpoint_dir=args.checkpoint_dir)
