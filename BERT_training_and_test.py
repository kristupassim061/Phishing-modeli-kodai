import pandas as pd
import torch
 
# Data splitting
from sklearn.model_selection import train_test_split
 
# Transformers
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
 
# Metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
 
# Confusion matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
 
 
# 1. LOAD DATA (tik realus duomenys)
 
df = pd.read_csv(r"C:\Bakis\phishing_email.csv", encoding="latin1")
df = df[['text_combined', 'label']].dropna()
df = df.sample(n=15000, random_state=42).reset_index(drop=True)
 
df['label'] = df['label'].astype(int)
df['text_combined'] = df['text_combined'].apply(lambda x: str(x).lower())
 
print(f"Dataset size: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts()}\n")
 
 
# 2. TRAIN / VAL / TEST SPLIT
 
train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
    df['text_combined'].tolist(),
    df['label'].tolist(),
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)
 
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_val_texts,
    train_val_labels,
    test_size=0.125,
    random_state=42,
    stratify=train_val_labels
)
 
print(f"Train size: {len(train_texts)}")
print(f"Val size:   {len(val_texts)}")
print(f"Test size:  {len(test_texts)}\n")
 
 
# 3. TOKENIZATION
 
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
 
train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=256)
val_encodings   = tokenizer(val_texts,   truncation=True, padding=True, max_length=256)
test_encodings  = tokenizer(test_texts,  truncation=True, padding=True, max_length=256)
 
 
# 4. DATASET CLASS
 
class EmailDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
 
    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item
 
    def __len__(self):
        return len(self.labels)
 
 
train_dataset = EmailDataset(train_encodings, train_labels)
val_dataset   = EmailDataset(val_encodings,   val_labels)
test_dataset  = EmailDataset(test_encodings,  test_labels)
 
 
# 5. LOAD MODEL
 
import os
 
model_path = r"C:\Bakis\BERT_model"
 
if os.path.exists(model_path):
    print("Loading existing model (fine-tuning)...")
    model = BertForSequenceClassification.from_pretrained(model_path)
else:
    print("No saved model found, loading base BERT...")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2
    )
 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("Using device:", device)
 
 
# 6. METRICS
 
def compute_metrics(pred):
    labels = pred.label_ids
    preds  = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy':  acc,
        'f1':        f1,
        'precision': precision,
        'recall':    recall
    }
 
 
# 7. TRAINING SETTINGS
 
training_args = TrainingArguments(
    output_dir='./results',
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    report_to="none"
)
 
 
# 8. TRAINER
 
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics
)
 
 
# 9. TRAIN
 
trainer.train()
 
# SAVE
 
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)
 
 
# 10. EVALUATE
 
results = trainer.evaluate(test_dataset)
print("\nTest Results:")
for k, v in results.items():
    print(f"  {k}: {round(v, 4)}")
 
 
# 11. SAMPLE PREDICTIONS
 
predictions = trainer.predict(test_dataset)
preds = predictions.predictions.argmax(-1)
 
print("\nSample Predictions vs Actual:")
for i in range(10):
    match = "✓" if preds[i] == test_labels[i] else "✗"
    print(f"  [{match}] Predicted: {preds[i]} | Actual: {test_labels[i]}")
 
 
# 12. CONFUSION MATRIX
 
y_true = predictions.label_ids
y_pred = np.argmax(predictions.predictions, axis=1)
 
cm = confusion_matrix(y_true, y_pred)
 
disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["SAFE", "PHISHING"]
)
 
disp.plot(cmap="Blues")
plt.title("BERT Confusion Matrix")
plt.tight_layout()
plt.show()
