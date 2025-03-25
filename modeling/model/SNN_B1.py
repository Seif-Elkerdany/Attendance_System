import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
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
from SiameseDataset import SiameseDataset
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")


class ViT_finetune(nn.Module):
    """
    Fine-tunes a pre-trained ViT-B-16 model for face embeddings.
    
    This model:
      - Accepts images of shape (B, 3, 112, 112) and upsamples them to 224x224 (ViT-B-16 pre-training size).
      - Removes the classification head.
      - Projects the resulting 768-dimensional feature vector to a 128-dimensional embedding.
    
    Parameters:
      embedding_dim (int): The dimensionality of the output embeddings (default: 128).
    """
    def __init__(self, embedding_dim=128):
        super(ViT_finetune, self).__init__()
        self.vit = models.vit_b_16(pretrained=True)
        # Depending on your torchvision version, the classification head might be named 'heads' or 'head'.
        self.vit.heads = nn.Identity()
        self.embedding = nn.Linear(768, embedding_dim)
        
    def forward(self, x):
        # x: (B, 3, 112, 112)
        x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        features = self.vit(x)            # (B, 768)
        embedding = self.embedding(features)  # (B, embedding_dim)
        return embedding

class SiameseNetwork(nn.Module):
    """
    Siamese Network that uses a shared fine-tuned face model to extract embeddings from two images.
    It computes the cosine similarity between the two embeddings.
    """
    
    def __init__(self, base_net):
        super(SiameseNetwork, self).__init__()
        self.base_net = base_net
        
    def forward(self, input1, input2):
        output1 = self.base_net(input1)
        output2 = self.base_net(input2)
        cosine_sim = F.cosine_similarity(output1, output2)
        return output1, output2, cosine_sim

class MarginCosineLoss(nn.Module):
    """
    Implements a margin cosine loss for Siamese networks.
    
    For similar pairs (label == 1): the loss is (1 - cosine_sim)^2.
    For dissimilar pairs (label == 0): the loss is (max(0, cosine_sim - margin))^2.
    
    The final loss is averaged over the batch.
    
    Parameters:
      margin (float): The margin value for dissimilar pairs (default: 0.35).
    """
    def __init__(self, margin=0.35):
        super(MarginCosineLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        cosine_sim = F.cosine_similarity(output1, output2)
        loss_pos = label * torch.pow(1 - cosine_sim, 2)
        loss_neg = (1 - label) * torch.pow(torch.clamp(cosine_sim - self.margin, min=0.0), 2)
        loss = loss_pos + loss_neg
        return torch.mean(loss)


def train_siamese_network(train_dataloader,
                          val_dataloader,
                          num_epochs=20,
                          learning_rate=0.001,
                          num_train_batches=100,
                          num_val_batches=20,
                          early_stopping_patience=3,
                          classification_threshold=0.5,
                          margin=0.35,
                          run_dir="runs/siamese_vit",
                          checkpoint_dir="checkpoints"):
    """
    Trains a Siamese network for face verification using a fine-tuned ViT-B-16 backbone and a margin cosine loss.
    
    Parameters:
      train_dataloader (DataLoader): Dataloader for training data.
      val_dataloader (DataLoader): Dataloader for validation data.
      num_epochs (int): Maximum number of epochs for training.
      learning_rate (float): Initial learning rate.
      num_train_batches (int): Maximum number of training iterations per epoch.
      num_val_batches (int): Maximum number of validation iterations per epoch.
      early_stopping_patience (int): Number of epochs with no improvement on validation loss before stopping.
      classification_threshold (float): Threshold on cosine similarity to classify pairs as similar (1) or dissimilar (0).
      margin (float): Margin parameter for the MarginCosineLoss.
      run_dir (str): Directory for TensorBoard logs.
      checkpoint_dir (str): Directory to save model checkpoints and confusion matrix plots.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("You are now training on", device)

    writer = SummaryWriter(run_dir)
    
    base_net = ViT_finetune(embedding_dim=128).to(device)
    model = SiameseNetwork(base_net).to(device)
    criterion = MarginCosineLoss(margin=margin)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss_epoch = 0.0
        train_batches = 0
        
        # Training loop using the train_dataloader.
        train_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Training")
        for batch_idx, (img1, img2, labels) in enumerate(train_bar):
            # Limit training to num_train_batches if specified.
            if batch_idx >= num_train_batches:
                break

            img1 = img1.to(device)
            img2 = img2.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            out1, out2, cosine_sim = model(img1, img2)
            loss = criterion(out1, out2, labels)
            loss.backward()
            optimizer.step()
            
            train_loss_epoch += loss.item()
            train_bar.set_postfix(loss=loss.item())
            train_batches += 1
            
        avg_train_loss = train_loss_epoch / train_batches if train_batches > 0 else 0.0
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        
        # Validation loop.
        model.eval()
        val_loss_epoch = 0.0
        all_preds = []
        all_labels = []
        val_batches = 0
        with torch.no_grad():
            val_bar = tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Validation")
            for batch_idx, (img1, img2, labels) in enumerate(val_bar):
                if batch_idx >= num_val_batches:
                    break

                img1 = img1.to(device)
                img2 = img2.to(device)
                labels = labels.to(device)
                
                out1, out2, cosine_sim = model(img1, img2)
                loss = criterion(out1, out2, labels)
                val_loss_epoch += loss.item()
                val_bar.set_postfix(loss=loss.item())
                
                # Classification: predict similar if cosine_sim > threshold.
                preds = (cosine_sim > classification_threshold).float().cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
                val_batches += 1
                
        avg_val_loss = val_loss_epoch / val_batches if val_batches > 0 else 0.0
        writer.add_scalar("Loss/Validation", avg_val_loss, epoch)
        print(f"Epoch {epoch+1} - Avg Train Loss: {avg_train_loss:.4f}, Avg Val Loss: {avg_val_loss:.4f}")
        
        # Compute classification metrics on the validation set.
        cm = confusion_matrix(all_labels, all_preds)
        acc = accuracy_score(all_labels, all_preds)
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        writer.add_scalar("Metrics/Accuracy", acc, epoch)
        writer.add_scalar("Metrics/Precision", prec, epoch)
        writer.add_scalar("Metrics/Recall", rec, epoch)
        writer.add_scalar("Metrics/F1", f1, epoch)
        
        print(f"Validation Metrics: Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}")
        
        # Plot confusion matrix heatmap.
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix - Epoch {epoch+1}")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        cm_path = os.path.join(checkpoint_dir, f"confusion_matrix_epoch_{epoch+1}.png")
        plt.savefig(cm_path)
        plt.close()

        # Log confusion matrix image to TensorBoard.
        cm_img = torch.tensor(plt.imread(cm_path))
        cm_img = cm_img.permute(2, 0, 1)  # Convert HWC to CHW.
        writer.add_image("Confusion_Matrix", cm_img, epoch)
        
        # Update learning rate scheduler and early stopping.
        scheduler.step(avg_val_loss)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            checkpoint_path = os.path.join(checkpoint_dir, f"best_model_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Model improved. Checkpoint saved to {checkpoint_path}")
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")
            
        if epochs_no_improve >= early_stopping_patience:
            print("Early stopping triggered.")
            break
            
    writer.close()


if __name__ == '__main__':

    # M7dsh y5af da bs 3l4an lw 3aizen t3mlo run mn el terminal xD
    # Argument Parser for Terminal Execution
    parser = argparse.ArgumentParser(description="Train a Siamese network with a ViT-B-16 backbone for face verification.")
    parser.add_argument("--batch_size", type=int, default=16, help="Number of image pairs per batch.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for dataloader.")
    parser.add_argument("--num_epochs", type=int, default=20, help="Maximum number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=0.001, help="Initial learning rate.")
    parser.add_argument("--num_train_batches", type=int, default=100, help="Training iterations per epoch.")
    parser.add_argument("--num_val_batches", type=int, default=20, help="Validation iterations per epoch.")
    parser.add_argument("--early_stopping_patience", type=int, default=3, help="Epochs to wait before early stopping.")
    parser.add_argument("--classification_threshold", type=float, default=0.5, help="Threshold on cosine similarity for classification.")
    parser.add_argument("--margin", type=float, default=0.35, help="Margin for MarginCosineLoss.")
    parser.add_argument("--run_dir", type=str, default="modeling/model/runs/siamese_vit", help="TensorBoard log directory.")
    parser.add_argument("--checkpoint_dir", type=str, default="modeling/model/checkpoints", help="Directory to save model checkpoints and plots.")
    
    args = parser.parse_args()

    train_df = pd.read_csv("/home/seif_elkerdany/projects/data/train_dataset.csv")
    test_df = pd.read_csv("/home/seif_elkerdany/projects/data/test_dataset.csv")

    train_df, val_df = train_test_split(test_df, test_size=0.5, random_state=42)

    train_dataset = SiameseDataset(train_df, train=True)
    val_dataset = SiameseDataset(val_df, train= False)
    
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    train_siamese_network(train_dataloader=train_dataloader,
                          val_dataloader=val_dataloader,
                          num_epochs=args.num_epochs,
                          learning_rate=args.learning_rate,
                          num_train_batches=args.num_train_batches,
                          num_val_batches=args.num_val_batches,
                          early_stopping_patience=args.early_stopping_patience,
                          classification_threshold=args.classification_threshold,
                          margin=args.margin,
                          run_dir=args.run_dir,
                          checkpoint_dir=args.checkpoint_dir)
    