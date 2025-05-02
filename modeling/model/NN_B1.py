import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import argparse
import warnings
from torch.utils.data import DataLoader
import pandas as pd
from SiameseDataset import SiameseDataset
from sklearn.metrics import roc_curve, roc_auc_score
from facenet_pytorch import InceptionResnetV1
warnings.filterwarnings("ignore")

class InceptionResnetBackbone(nn.Module):
    def __init__(self, pretrained='vggface2', embedding_dim=512):
        super().__init__()
        self.model = InceptionResnetV1(pretrained=pretrained)

        self.embedding_dim = embedding_dim
        if embedding_dim != 512:
            self.proj = nn.Linear(512, embedding_dim)
        else:
            self.proj = None

    def forward(self, x):
        x = self.model(x)  # (B, 512)
        if self.proj:
            x = self.proj(x)
        x = F.normalize(x, p=2, dim=1)  # L2 normalization
        return x
    

class SiameseClassifier(nn.Module):
    def __init__(self, embedding_dim=512, hidden_dim=256, pretrained='vggface2'):
        super().__init__() 

        self.backbone = InceptionResnetBackbone(pretrained, embedding_dim)
        
        # classifier head on [|e1−e2|, e1*e2]
        self.classifier = nn.Sequential(
            nn.Linear(2 * embedding_dim, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(0.45),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, img1, img2):
        """
        img1, img2: tensors (B, C, H, W)
        returns:
          logits: (B,) raw scores before sigmoid
        """
        e1 = self.backbone(img1)  # (B, D)
        e2 = self.backbone(img2)  # (B, D)

        x = torch.cat([e1, e2], dim=1)  # (B, 2D)

        logits = self.classifier(x).squeeze(1)  # (B,)
        return logits

    @torch.no_grad()
    def predict(self, img1, img2, threshold=0.5):
        """
        Runs a forward pass, applies sigmoid, and thresholds.
        Returns:
          probs: (B,) in [0,1]
          preds: (B,) bool
        """
        logits = self.forward(img1, img2)
        probs = torch.sigmoid(logits)
        return probs, probs >= threshold

def train_siamese_network(train_dataloader,
                          val_dataloader,
                          num_epochs=30,
                          pos_weight=None,
                          early_stopping_patience=3,
                          classification_threshold=0.5,
                          embedding_dim=512,
                          run_dir="runs/siamese_B3_net",
                          checkpoint_dir="checkpoints"):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("You are now training on", device)

    writer = SummaryWriter(run_dir)

    model = SiameseClassifier(
        embedding_dim=embedding_dim,
        hidden_dim=256,
        pretrained='vggface2'
    ).to(device)

    # Freeze everything except the last Inception block
    for name, param in model.backbone.model.named_parameters():
        if not any(k in name for k in ["block8", "block7"]):   
            param.requires_grad = False

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    
    backbone_finetune = [p for p in model.backbone.parameters() if p.requires_grad]
    head_params  = list(model.classifier.parameters())
    optimizer = optim.AdamW(
                    [
                        {"params": head_params,       "lr": 1e-4},
                        {"params": backbone_finetune, "lr": 1e-5},
                    ],
                    weight_decay=1e-2
                )

    # cosine annealing schedule
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2)

    best_f1 = 0.0 
    epochs_no_improve = 0
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(1, num_epochs+1):
        # ——— Training ———
        model.train()
        train_loss = 0.0
        for batch_idx, (img1, img2, labels) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch}/{num_epochs} Train")):

            img1, img2 = img1.to(device), img2.to(device)
            labels = labels.to(device).float()

            optimizer.zero_grad()
            logits = model(img1, img2)                     
            loss   = criterion(logits, labels)             

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_dataloader)
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)

        # ——— Validation ———
        model.eval()
        val_loss = 0.0
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for batch_idx, (img1, img2, labels) in enumerate(tqdm(val_dataloader, desc=f"Epoch {epoch}/{num_epochs} Val")):

                img1, img2 = img1.to(device), img2.to(device)
                labels = labels.to(device).float()

                logits = model(img1, img2)
                loss   = criterion(logits, labels)
                val_loss += loss.item()

                probs = torch.sigmoid(logits)
                preds = (probs > classification_threshold).long()

                all_probs.extend(probs.cpu().tolist())
                all_preds.extend(preds.cpu().tolist())
                all_labels.extend(labels.cpu().long().tolist())

        avg_val_loss = val_loss / len(val_dataloader)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        print(f"\nEpoch {epoch} — Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # ——— Metrics & Checkpointing ———
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
        fpr, tpr, _ = roc_curve(all_labels, all_probs, pos_label=1)
        roc_auc = roc_auc_score(all_labels, all_probs)
        fig = plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
        plt.plot([0,1], [0,1], 'r--')
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve")

        # log to TensorBoard
        writer.add_figure("ROC_Curve", fig, epoch)
        fig.savefig(os.path.join(checkpoint_dir, f"roc_epoch_{epoch}.png"))
        plt.close(fig)

        scheduler.step(epoch)

        if f1 > best_f1:
            best_f1 = f1
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict()
            }, os.path.join(checkpoint_dir, f"checkpoint_epoch{epoch}.pt"))
            print("Model improved; checkpoint saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= early_stopping_patience:
                print("Early stopping.")
                break

    writer.close()


if __name__ == '__main__':
    import random
    import numpy as np

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

    parser = argparse.ArgumentParser(description="Train a Siamese network for face verification for our Attendace System.")
    parser.add_argument("--batch_size", type=int, default=32, help="Number of image pairs per batch.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader.")
    parser.add_argument("--num_epochs", type=int, default=30, help="Maximum number of training epochs.")
    parser.add_argument("--early_stopping_patience", type=int, default=4, help="Epochs to wait before early stopping.")
    parser.add_argument("--classification_threshold", type=float, default=0.5, help="Probability threshold for classifying a pair as positive.")
    parser.add_argument("--run_dir", type=str, default="/home/seif_elkerdany/projects/modeling/model/runs/siamese_B1_net", help="TensorBoard log directory.")
    parser.add_argument("--checkpoint_dir", type=str, default="/home/seif_elkerdany/projects/modeling/model/checkpoints/B1", help="Directory to save model checkpoints and plots.")
    
    args = parser.parse_args()

    train_df = pd.read_csv("/home/seif_elkerdany/projects/data/train_dataset_2.csv")
    
    pos = train_df['label'].sum()
    neg = len(train_df) - pos
    pos_weight = torch.tensor([neg / (pos + 1e-6)])  

    val_df = pd.read_csv("/home/seif_elkerdany/projects/data/val_split_2.csv")

    train_dataset = SiameseDataset(train_df, train=True)
    val_dataset = SiameseDataset(val_df, train= False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    train_siamese_network(train_dataloader=train_dataloader,
                          val_dataloader=val_dataloader,
                          num_epochs=args.num_epochs,
                          early_stopping_patience=args.early_stopping_patience,
                          classification_threshold=args.classification_threshold,
                          pos_weight=pos_weight,
                          run_dir=args.run_dir,
                          checkpoint_dir=args.checkpoint_dir)
