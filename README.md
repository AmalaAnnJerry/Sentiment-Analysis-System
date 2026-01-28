# Sentiment Analysis System using NLP

An end-to-end sentiment analysis pipeline to classify movie reviews as **positive** or **negative** using classical Natural Language Processing (NLP) techniques and machine learning. This project demonstrates how raw text data can be **cleaned, transformed, modeled, evaluated**, and **used for real-world inference**.

---

## Problem Statement

Online platforms receive a massive number of user-generated text reviews. Manually analyzing sentiment is time-consuming and subjective. This project aims to build an automated sentiment classification system that:

- Processes raw text reviews
- Learns sentiment patterns from labeled data
- Predicts sentiment for unseen real-world reviews

---

## Dataset

This project uses the [IMDB Movie Reviews dataset](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) from Kaggle.

**Note:** The dataset is **not included** in this repository due to size constraints.  
Please download it and place it in the `data/` folder before running `train_model.py`.

---

## Methodology

### 1. Data Preprocessing
- Convert text to lowercase  
- Remove HTML tags and special characters  
- Remove stopwords  
- Apply **lemmatization** to normalize words while preserving meaning  

### 2. Feature Engineering
- Use **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert text into numerical vectors  
- Limit vocabulary to **top 5,000 terms** to reduce noise and computational cost  

### 3. Model Training
- Train a **Logistic Regression** classifier on TF-IDF features  
- Use **80% of the dataset** for training  
- Increase iterations to ensure proper convergence  

### 4. Model Evaluation
- Metrics used: **accuracy, precision, recall, F1-score, confusion matrix**  
- Balanced evaluation for **both positive and negative sentiment classes**  

### 5. Decision Threshold Tuning
- Predicted probabilities are used to adjust the **decision threshold**  
- Improves classification of nuanced and mixed-sentiment reviews  

### 6. Model Persistence
- Save the trained model and TF-IDF vectorizer using **joblib**  
- Enables reusing the model for inference without retraining  

---

## Results

- **Accuracy:** 88%  
- **Balanced precision, recall, and F1-score** across both classes  
- Generalizes well to **unseen, real-world reviews**

---

## Tech Stack

- **Python**  
- **scikit-learn**  
- **NLTK**  
- **pandas**  
- **NumPy**  
- **joblib**

---

## Usage

### 1. Install dependencies
pip install -r requirements.txt
### 2. Prepare dataset
Download IMDB dataset from Kaggle
Place IMDB Dataset.csv inside the data/ folder
### 3. Train the model
python train_model.py
### 4. Make predictions
python predict.py
