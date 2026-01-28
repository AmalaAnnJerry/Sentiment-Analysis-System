# Sentiment Analysis System using NLP

This project implements an end-to-end sentiment analysis pipeline to classify movie reviews as positive or negative using classical Natural Language Processing techniques and machine learning. The project demonstrates how raw text data can be cleaned, transformed, modeled, evaluated, and used for real-world inference.

**Problem Statement**
Online platforms receive a massive number of user-generated text reviews. Manually analyzing sentiment is time-consuming and subjective. This project aims to build an automated sentiment classification system that processes raw text reviews, learns sentiment patterns from labeled data, and predicts sentiment for unseen real-world reviews.

**Dataset**
IMDB Movie Reviews Dataset
The dataset contains 50,000 movie reviews evenly split between positive and negative classes. The reviews are long-form, real-world text, making the dataset suitable for practical sentiment analysis tasks.

**Methodology**
1. Data Preprocessing
Raw text reviews are preprocessed by converting text to lowercase, removing HTML tags and special characters, removing stopwords, and applying lemmatization to normalize words while preserving meaning.

2. Feature Engineering
TF-IDF (Term Frequency–Inverse Document Frequency) is used to convert text into numerical feature vectors. The vocabulary size is limited to the top 5,000 most important terms to reduce noise and computational cost.

3. Model Training
A Logistic Regression classifier is trained on the TF-IDF features. The model is trained using 80 percent of the dataset, with increased iterations to ensure proper convergence.

4. Model Evaluation
The model is evaluated using accuracy, precision, recall, F1-score, and a confusion matrix. These metrics provide a balanced understanding of the model’s performance across both sentiment classes.

5. Decision Threshold Tuning
Instead of relying only on default hard predictions, predicted probabilities are used to adjust the decision threshold. This improves classification for nuanced and mixed-sentiment reviews.

6. Model Persistence
The trained model and TF-IDF vectorizer are saved using joblib, allowing the model to be reused during inference without retraining.

**Results**
The model achieves an accuracy of 88 percent with balanced precision, recall, and F1-score across positive and negative classes. It generalizes well to unseen, real-world reviews outside the training dataset.

**Tech Stack**
Python
scikit-learn
NLTK
pandas
NumPy
joblib