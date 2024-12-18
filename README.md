# Malicious URL Detector

#### Project Overview

The Malicious URL Detector is a machine learning-based project designed to classify URLs as malicious or benign. It leverages natural language processing (NLP) techniques to tokenize URLs into features and uses machine learning algorithms like Logistic Regression and Support Vector Machines (SVM) for classification.

Our goal is to identify potentially harmful URLs to enhance cybersecurity defenses.

#### Tools and Libraries Used

**Programming Language:** 
Python

**Libraries and Frameworks:**
- Pandas: For data manipulation and preprocessing.
- NumPy: For numerical computations.
- Scikit-learn: For machine learning models, feature extraction, and evaluation.
- Logistic Regression: Linear classifier for baseline modeling.
- Support Vector Machines (SVM): Non-linear classifier for enhanced accuracy.
- TfidfVectorizer: To convert URLs into numerical feature vectors based on term frequency-inverse document frequency.
- CountVectorizer: Alternative feature extraction method (not used directly in the final model).

**Dataset**
Source: malicious_phish.csv _(credit: Kaggle)_
https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset/data?select=malicious_phish.csv

This dataset contains labeled data with columns:
url: The actual URL to be analyzed.
type: The label for the URL type, such as "benign" or "malicious".

The dataset is shuffled randomly to avoid bias during training.

**Methodology**
URLs are tokenized using the custom function makeTokens.

**Tokenization Steps:**
- Split the URL by / (slashes) and then by - (dashes) and . (dots).
- Combine all tokens into a set to remove duplicates.
- Remove frequent and redundant tokens like "com" and "https://www." to - enhance feature relevance.
- Features are vectorized using TfidfVectorizer, converting textual data into a numerical matrix.

**Feature Engineering:**
The makeTokens function generates meaningful tokens from URLs to serve as input features.

**TfidfVectorizer:**
Calculate the importance of tokens within the dataset.
Normalize feature values to account for varying token frequencies.

**Train-Test Split:**
Data is split into training and testing subsets (80:20 ratio) using train_test_split.
A random seed (random_state=434) ensures reproducibility.

**Model Building:**
Logistic Regression:
A linear classification model for quick and interpretable results.
Support Vector Machines (SVM):
A non-linear classifier with radial basis function (RBF) kernel (gamma='scale').
Hyperparameters:
C=79: Regularization parameter.
random_state=12345: Ensures consistent results.

**Model Evaluation:**
Models are evaluated based on accuracy on the test set.

**Key Results**
Logistic Regression Accuracy: Prints the accuracy score on the test set.
SVM Accuracy: Achieves a higher score compared to Logistic Regression, demonstrating its ability to capture complex patterns in URL tokens.

**Future Improvements**
1. Expand Dataset: Add more labeled data for diverse and robust model training.
2. Feature Selection: Explore advanced feature engineering techniques to improve tokenization and relevance.
3. Deploy Model: Convert the script into a web or API-based tool for real-world applications.
