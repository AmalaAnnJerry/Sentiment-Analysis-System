import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import nltk
import joblib
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

nltk.download('wordnet')
nltk.download('omw-1.4')  # optional for wordnet

df = pd.read_csv(r"C:\Users\hp\OneDrive\Documents\Desktop\sentiment-analysis\data\IMDB Dataset.csv")

# Basic info
print("Dataset shape:", df.shape)
print("\nColumn names:", df.columns)

# Check class distribution
print("\nSentiment counts:")
print(df['sentiment'].value_counts())

# Preview data
print("\nSample reviews:")
print(df.head(3))

def basic_clean(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    return text

print("\nAfter basic cleaning:\n")
print(basic_clean(df['review'][0])[:500])

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    return ' '.join(filtered_words)

cleaned = basic_clean(df['review'][0])
print("\nAfter stopword removal:\n")
print(remove_stopwords(cleaned)[:500])

lemmatizer = WordNetLemmatizer()
def lemmatize_text(text):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)

cleaned = basic_clean(df['review'][0])
no_stopwords = remove_stopwords(cleaned)
lemmatized = lemmatize_text(no_stopwords)

print("\nAfter lemmatization:\n")
print(lemmatized[:500])

# Prepare labels
y = df['sentiment']  # 1 = positive, 0 = negative

# Prepare features: cleaned + lemmatized reviews
df['processed_review'] = df['review'].apply(lambda x: lemmatize_text(remove_stopwords(basic_clean(x))))
X = df['processed_review']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2)
)

# Fit on training data
X_train_tfidf = tfidf.fit_transform(X_train)

# Transform test data
X_test_tfidf = tfidf.transform(X_test)

print("X_train size:", X_train.shape)
print("X_test size:", X_test.shape)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_tfidf, y_train)
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
print(cm)


import joblib

joblib.dump(model, "sentiment_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")

print("Model and vectorizer saved successfully!")








