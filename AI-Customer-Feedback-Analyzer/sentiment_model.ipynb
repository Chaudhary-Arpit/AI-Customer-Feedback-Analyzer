# Sentiment Model Notebook (outline)
# - Load cleaned_feedback.csv
# - Create labels (positive/negative/neutral) for training
# - Train a DistilBERT model using Hugging Face Trainer (or use a simple sklearn baseline)
# - Save the model and tokenizer

# NOTE: Full training requires GPU and time; here we provide starter code.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib

df = pd.read_csv('cleaned_feedback.csv')
# Simple labeling heuristic for demo purposes
def label_text(x):
    txt = str(x).lower()
    if any(w in txt for w in ['excellent','amazing','great','satisfied','helpful','fast']):
        return 'positive'
    if any(w in txt for w in ['terrible','bad','poor','frustrat','missing','damag','crash']):
        return 'negative'
    return 'neutral'

df['label'] = df['cleaned_feedback'].apply(label_text)

X = df['cleaned_feedback']
y = df['label']
vec = TfidfVectorizer()
Xv = vec.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(Xv, y, test_size=0.2, random_state=42)

clf = LogisticRegression(max_iter=500)
clf.fit(X_train, y_train)
pred = clf.predict(X_test)
print(classification_report(y_test, pred))

joblib.dump(clf, 'sentiment_model.pkl')
joblib.dump(vec, 'tfidf_vectorizer.pkl')
print('Saved sentiment_model.pkl and tfidf_vectorizer.pkl')
