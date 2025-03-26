import torch
from torch.utils.data import DataLoader
from SNN_B1 import CNNBackbone, SiameseNetwork
from SiameseDataset import SiameseDataset  
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

embedding_dim = 1024
backbone = CNNBackbone(embedding_dim=embedding_dim)
model = SiameseNetwork(base_net=backbone)
model = model.to(device)


checkpoint_path = "/home/seif_elkerdany/projects/modeling/model/checkpoints/B1/best_model_epoch_19.pt"  
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()


test_df = pd.read_csv("/home/seif_elkerdany/projects/data/test_split.csv")
test_dataset = SiameseDataset(test_df, train=False)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)


all_preds, all_labels, all_sims = [], [], []
with torch.no_grad():
    for img1, img2, labels in test_dataloader:
        img1 = img1.to(device)
        img2 = img2.to(device)
        labels = labels.to(device)
        _, _, cosine_sim = model(img1, img2)
        all_sims.extend(cosine_sim.cpu().numpy())
        preds = (cosine_sim > 0.5).float().cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

acc = accuracy_score(all_labels, all_preds)
prec = precision_score(all_labels, all_preds, zero_division=0)
rec = recall_score(all_labels, all_preds, zero_division=0)
f1 = f1_score(all_labels, all_preds, zero_division=0)
cm = confusion_matrix(all_labels, all_preds)

print("Accuracy:", acc)
print("Precision:", prec)
print("Recall:", rec)
print("F1 Score:", f1)
print("Confusion Matrix:")
print(cm)