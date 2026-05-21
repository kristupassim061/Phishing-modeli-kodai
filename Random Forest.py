import pandas as pd

# ML tools
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# 1. LOAD DATA

df = pd.read_csv(r"C:\Bakis\phishing_email.csv", encoding="latin1")

df = df.sample(n=15000, random_state=42).reset_index(drop=True)
df.dropna(inplace=True)

df['label'] = df['label'].astype(int)
df['text_combined'] = df['text_combined'].str.lower()

# 2. SPLIT DATA

X_train, X_test, y_train, y_test = train_test_split(
    df['text_combined'],
    df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# 3. TEXT to NUMBERS (TF-IDF)

vectorizer = TfidfVectorizer(max_features=5000)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 4. TRAIN RANDOM FOREST

model = RandomForestClassifier(
    n_estimators=100,   # number of trees
    max_depth=None,
    random_state=42,
    n_jobs=-1           # use all CPU cores
)

model.fit(X_train_vec, y_train)

# 5. EVALUATE

y_pred = model.predict(X_test_vec)

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

cm = confusion_matrix(y_test, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["SAFE", "PHISHING"]
)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nReport:\n", classification_report(y_test, y_pred))

disp.plot(cmap="Blues")
plt.title("Random Forest Confusion Matrix")
plt.show()