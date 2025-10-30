
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

df = pd.read_csv('sample_feedback.csv')
df = df.drop_duplicates()
df['feedback'] = df['feedback'].fillna('')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text)).lower()
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(tokens)

df['cleaned_feedback'] = df['feedback'].apply(clean_text)
df.to_csv('cleaned_feedback.csv', index=False)
print('Saved cleaned_feedback.csv')
