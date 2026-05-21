import pandas as pd
import torch
import numpy as np

from transformers import BertTokenizer
from transformers import BertForSequenceClassification

from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset, DataLoader


# =========================
# 1. LOAD FULL TEST DATASET
# =========================

test_df = pd.read_csv(r"C:\Bakis\phishing_dataset_15000.csv")
test_df = test_df[['text_combined', 'label']].dropna()
test_df['text_combined'] = test_df['text_combined'].astype(str).str.lower()
test_df['label'] = test_df['label'].astype(int)

print(f"Test dataset size: {len(test_df)}")
print(f"Label distribution:\n{test_df['label'].value_counts()}\n")


# =========================
# 2. LOAD SAVED MODEL
# =========================

model_path = r"C:\Bakis\BERT_model"

tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("Using device:", device)


# =========================
# 3. TOKENIZATION
# =========================

encodings = tokenizer(
    test_df['text_combined'].tolist(),
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors="pt"
)


# =========================
# 4. DATALOADER
# =========================

dataset = TensorDataset(
    encodings['input_ids'],
    encodings['attention_mask']
)

loader = DataLoader(dataset, batch_size=16)


# =========================
# 5. PREDICTIONS
# =========================

all_preds = []

with torch.no_grad():
    for batch in loader:
        input_ids, attention_mask = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())


# =========================
# 6. RESULTS
# =========================

y_true = test_df['label'].values
y_pred = np.array(all_preds)

print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=["SAFE", "PHISHING"]))


# =========================
# 7. CONFUSION MATRIX
# =========================

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["SAFE", "PHISHING"]
)

disp.plot(cmap="Blues")
plt.title("BERT - Cross Dataset Test")
plt.tight_layout()
plt.show()