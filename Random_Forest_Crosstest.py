import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt


# 1. TRAIN DATASET

train_df = pd.read_csv(
    r"C:\Bakis\phishing_email.csv",
    encoding="latin1"
)

train_df = train_df[['text_combined', 'label']].dropna()

train_df['text_combined'] = train_df['text_combined'].astype(str).str.lower()
train_df['label'] = train_df['label'].astype(int)

print("TRAIN dataset size:", len(train_df))


# 2. TEST DATASET

test_df = pd.read_csv(
    r"C:\Bakis\phishing_dataset_15000.csv"
)

test_df = test_df[['text_combined', 'label']].dropna()

test_df['text_combined'] = test_df['text_combined'].astype(str).str.lower()
test_df['label'] = test_df['label'].astype(int)

print("TEST dataset size:", len(test_df))


# 3. TF-IDF

vectorizer = TfidfVectorizer(max_features=5000)

# FIT ONLY ON TRAIN
X_train = vectorizer.fit_transform(train_df['text_combined'])

# TRANSFORM TEST
X_test = vectorizer.transform(test_df['text_combined'])

y_train = train_df['label']
y_test = test_df['label']


# 4. RANDOM FOREST

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)


# 5. PREDICT

y_pred = model.predict(X_test)


# 6. RESULTS

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))


# 7. CONFUSION MATRIX

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["SAFE", "PHISHING"]
)

disp.plot(cmap="Blues")

plt.title("Random Forest - Cross Dataset Test")

plt.show()