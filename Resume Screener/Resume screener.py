# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 11:00:26 2025

@author: shubham
"""

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import matplotlib.pyplot as plt
import seaborn as sns  

# Load the dataset
df = pd.read_csv("ResumeDataset.csv")

# Display the first few rows
print(df.head())

# Check dataset info
print(df.info())

# Check column names
print(df.columns)

# Download required NLTK data (run this once)
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load dataset
df = pd.read_csv("ResumeDataset.csv")

# Visualize category distribution
plt.figure(figsize=(8, 6))
sns.countplot(x='Category', data=df, order=df['Category'].value_counts().index)
plt.title("Category Distribution")
plt.xticks(rotation=45)
plt.show()

# Initialize NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_resume(text):
    """Cleans and preprocesses resume text."""
    text = re.sub(r'[^a-zA-Z]', ' ', text)  
    text = text.lower() 
    words = word_tokenize(text) 
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words and len(word) > 2]
    return " ".join(words)

# Apply cleaning function
df["Cleaned_Resume"] = df["Resume"].apply(clean_resume)

# Display sample cleaned text
print(df[["Resume", "Cleaned_Resume"]].head())

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000) 

# Fit and transform the cleaned resumes
X_resumes = tfidf_vectorizer.fit_transform(df["Cleaned_Resume"])

# Display the feature names (top words)
print(tfidf_vectorizer.get_feature_names_out()[:20])

# Convert the result to a DataFrame
df_tfidf = pd.DataFrame(X_resumes.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

# Show the transformed feature matrix
print(df_tfidf.head())

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Prepare the data
X = X_resumes 
y = df["Category"] 

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize and train Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import joblib

# Save the trained model
joblib.dump(model, 'resume_matcher_model.pkl')

# Save the TF-IDF vectorizer
joblib.dump(tfidf_vectorizer, 'tfidf_vectorizer.pkl')


