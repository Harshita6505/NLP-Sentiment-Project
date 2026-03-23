# ================== Q1: SENTIMENT ANALYSIS ==================

import pandas as pd
import nltk
import matplotlib.pyplot as plt

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Download NLTK data (run once)
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

# Load dataset
data = pd.read_csv("archive/IMDB Dataset.csv")

# Take 200 samples
data = data.sample(200, random_state=42)

texts = data['review']
labels = data['sentiment']

# Preprocessing function
def preprocess(text):
    tokens = word_tokenize(text.lower())
    tokens = [w for w in tokens if w.isalpha()]
    tokens = [w for w in tokens if w not in stopwords.words('english')]
    return " ".join(tokens)

# Apply preprocessing
texts = texts.apply(preprocess)

# TF-IDF Feature Extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, labels, test_size=0.3, stratify=labels, random_state=42
)

# Train Naïve Bayes model
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Evaluation
print("\n===== Q1: SENTIMENT ANALYSIS =====")
print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

print("\nClassification Report:\n", classification_report(y_test, y_pred))


# Graph for Accuracy
accuracy = accuracy_score(y_test, y_pred)

plt.figure()
plt.bar(['Naive Bayes'], [accuracy])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.show()


# ================== Q2: WORD EMBEDDING ==================

import gensim.downloader as api
from numpy import dot
from numpy.linalg import norm

print("\nLoading Word2Vec model (this may take time)...")

model_w2v = api.load("word2vec-google-news-300")

def cosine_similarity(a, b):
    return dot(a, b) / (norm(a) * norm(b))

pairs = [("king", "queen"), ("doctor", "nurse"), ("car", "tree")]

print("\n===== Q2: WORD EMBEDDING =====")

for w1, w2 in pairs:
    sim = cosine_similarity(model_w2v[w1], model_w2v[w2])
    print(f"{w1} - {w2}: {sim:.4f}")