import pandas as pd
import torch
import numpy as np

from transformers import BertTokenizer
from transformers import BertForSequenceClassification

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt


# =========================
# 1. LOAD DATA (tas pats kaip BERT.py ir Naive Bayes)
# =========================

df = pd.read_csv(r"C:\Bakis\phishing_email.csv", encoding="latin1")
df = df[['text_combined', 'label']].dropna()
df = df.sample(n=15000, random_state=42).reset_index(drop=True)

df['text_combined'] = df['text_combined'].astype(str).str.lower()
df['label'] = df['label'].astype(int)

print(f"Dataset size: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts()}\n")


# =========================
# 2. IŠSKIRIAM TIK TESTAVIMO RINKINĮ
# =========================

_, test_texts, _, test_labels = train_test_split(
    df['text_combined'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

print(f"Test size: {len(test_texts)}\n")


# =========================
# 3. LOAD SAVED MODEL
# =========================

model_path = r"C:\Bakis\BERT_model"

tokenizer = BertTokenizer.from_pretrained(model_path)

model = BertForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

print("Using device:", device)


# =========================
# 4. TOKENIZATION
# =========================

encodings = tokenizer(
    test_texts,
    truncation=True,
    padding=True,
    max_length=256,
    return_tensors="pt"
)


# =========================
# 5. DATALOADER
# =========================

from torch.utils.data import TensorDataset, DataLoader

dataset = TensorDataset(
    encodings['input_ids'],
    encodings['attention_mask']
)

loader = DataLoader(dataset, batch_size=16)


# =========================
# 6. PREDICTIONS
# =========================

all_preds = []

with torch.no_grad():
    for batch in loader:
        input_ids, attention_mask = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())


# =========================
# 7. RESULTS
# =========================

y_true = np.array(test_labels)
y_pred = np.array(all_preds)

print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=["SAFE", "PHISHING"]))


# =========================
# 8. CONFUSION MATRIX
# =========================

cm = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["SAFE", "PHISHING"]
)

disp.plot(cmap="Blues")
plt.title("BERT Confusion Matrix")
plt.tight_layout()
plt.show()