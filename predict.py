import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

# Load saved objects
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return " ".join(words)

# User input
review = input("Enter a movie review: ")

cleaned = clean_text(review)
vector = tfidf.transform([cleaned])

proba = model.predict_proba(vector)[0]
positive_prob = proba[1]

print("Positive probability:", positive_prob)

if positive_prob >= 0.45:
    print("Sentiment: POSITIVE 😊")
else:
    print("Sentiment: NEGATIVE 😞")

