import os
import time
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

# ========== Load Dataset ==========
df = pd.read_csv("Dataset/final_emails.csv")

# Clean and unify labels
df['label'] = df['label'].replace({
    "Ham": 0,
    "Spam": 1,
    "ham": 0,
    "spam": 1
})

# Ensure labels are int (not strings)
df['label'] = df['label'].astype(int)

print(" Unique labels after cleaning:", df['label'].unique())

# ========== Split ==========
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

# ========== Vectorization ==========
vectorizer = TfidfVectorizer(max_features=20000, stop_words="english")
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ========== Model ==========
model = LinearSVC(C=1.0, max_iter=5000)

print(" Training started...")
start = time.time()
model.fit(X_train_tfidf, y_train)
end = time.time()
print(f" Training completed in {end-start:.2f} seconds")

# ========== Evaluation ==========
y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print("\n Accuracy:", accuracy)
print("\n Classification Report:\n", classification_report(y_test, y_pred, target_names=["Ham", "Spam"]))

# ========== Save Model ==========
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/linear_svc_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")

print("\n Model and vectorizer saved in 'models/' folder")
