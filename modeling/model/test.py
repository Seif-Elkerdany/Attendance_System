import torch
from torch.utils.data import DataLoader
from .NN_B3 import SiameseClassifier
from .SiameseDataset import SiameseDataset  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve
)
import os

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SiameseClassifier(embedding_dim=256).to(device)
    ckpt_path = "/home/seif_elkerdany/projects/modeling/model/checkpoints/B3.1/checkpoint_epoch1.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    state_dict = ckpt.get("model_state_dict", ckpt)

    cleaned = {
        (k.replace("module.", "") if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    model.load_state_dict(cleaned)
    model.eval()


    test_df = pd.read_csv("/home/seif_elkerdany/projects/data/test_split.csv")
    test_df['ok1'] = test_df['image1'].apply(os.path.exists)
    test_df['ok2'] = test_df['image2'].apply(os.path.exists)
    missing = test_df[~(test_df.ok1 & test_df.ok2)]
    if not missing.empty:
        print("Warning: the following image pairs are missing on disk, they will be skipped:")
        print(missing[['image1','image2']])
    
    test_df = test_df[test_df.ok1 & test_df.ok2].drop(columns=['ok1','ok2'])

    test_dataset = SiameseDataset(test_df, train=False)
    test_dataloader = DataLoader(
        test_dataset, batch_size=16, shuffle=False, num_workers=4
    )

    all_sims, all_preds, all_labels = [], [], []
    with torch.no_grad():
        for img1, img2, labels in test_dataloader:

            img1, img2, labels = img1.to(device), img2.to(device), labels.to(device)

            logits = model(img1, img2)                # shape [B]
            probs  = torch.sigmoid(logits)            # shape [B], in [0,1]
            sims   = probs.cpu().numpy()             # for PR-curve

            all_sims.extend(sims)
            all_preds.extend((sims > 0.8953549861907959).astype(float))
            all_labels.extend(labels.cpu().numpy())

    precision, recall, thresholds = precision_recall_curve(all_labels, all_sims)
    f1_scores     = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-8)
    best_threshold = thresholds[np.argmax(f1_scores)]
    print(f"Best threshold (max F1): {best_threshold:.4f}")

    final_preds = (np.array(all_sims) > best_threshold).astype(int)
    acc   = accuracy_score(all_labels, final_preds)
    prec  = precision_score(all_labels, final_preds, zero_division=0)
    rec   = recall_score(all_labels, final_preds, zero_division=0)
    f1    = f1_score(all_labels, final_preds, zero_division=0)
    cm    = confusion_matrix(all_labels, final_preds)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, marker='.', label='PR curve')
    idx = np.argmax(f1_scores)
    plt.scatter(recall[idx], precision[idx], color='red',
                label=f'Best thresh={best_threshold:.3f}')
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.legend()
    plt.savefig("Thresholds_B3.1.png")
    plt.close()

    print("Accuracy:",  acc)
    print("Precision:", prec)
    print("Recall:",    rec)
    print("F1 Score:",  f1)
    print("Confusion Matrix:\n", cm)


def predict(model, image1, image2, threshold=0.7363):
    """
    Uses the built-in `model.predict()` which returns (probs, bool_preds).
    """
    model.eval()
    probs, bool_preds = model.predict(image1, image2, threshold)
    return bool_preds.cpu().numpy()


if __name__ == '__main__':
    main()
