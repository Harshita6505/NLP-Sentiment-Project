# NLP Sentiment Analysis & Word Embedding

## Project Description
This project is a comprehensive NLP system that performs sentiment analysis, emotion detection, and keyword extraction on movie reviews. It starts with a baseline Naïve Bayes model and progresses to advanced machine learning models like SVM with hyperparameter tuning. The project is structured modularly to support multiple NLP tasks.

## Features
-Sentiment Analysis (Naive Bayes, Logistic Regression, SVM)
-TF-IDF vectorization
-Hyperparameter tuning (GridSearchCV)
-Emotion detection from text
-Keyword extraction
-Word similarity using Word2Vec
-Visualization (confusion matrix, graphs)

## Results
- Baseline (Naive Bayes): ~73%
- Advanced (SVM): ~90%+ accuracy

## Technologies Used
- Python
- NLTK
- Scikit-learn
- Gensim
- Matplotlib

## How to Run
# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate   (or venv\Scripts\activate on Windows)

# Install dependencies
pip install -r requirements.txt

# Run baseline model
python sentiment.py

# Run advanced analysis
jupyter notebook
