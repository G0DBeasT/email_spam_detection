# Email Spam Detection and Classification

## Overview
This project demonstrates a complete machine learning pipeline for detecting and classifying spam emails. By leveraging word and character frequencies extracted from emails, the models classify each email as either spam or non-spam.

## Dataset
The dataset used is `spambase`, imported directly via `scikit-learn`'s `fetch_openml`.
- **Instances:** 4601
- **Features:** 57 continuous features representing frequencies of specific words and characters.
- **Target:** Binary label (`1` for spam, `0` for non-spam).

## Machine Learning Pipeline

1. **Data Loading:** 
   The dataset is loaded and combined into a `pandas` DataFrame for easy manipulation.

2. **Exploratory Data Analysis (EDA):**
   The distribution of the target variable is analyzed, revealing a split of approximately 60.6% non-spam and 39.4% spam emails.

3. **Data Preprocessing:**
   The data is split into training (80%) and testing (20%) sets. A `StandardScaler` is applied to standardize the features so they are on the same scale, which is crucial for many machine learning algorithms.

4. **Model Training:**
   Two distinct models are trained to compare their effectiveness:
   - **Gaussian Naive Bayes:** A fast probabilistic classifier that serves as a solid baseline for text/frequency data.
   - **Random Forest Classifier:** A robust ensemble method known for high performance on tabular datasets.

5. **Model Evaluation:**
   The models are evaluated using several metrics:
   - Accuracy
   - Precision
   - Recall
   - F1-Score
   
   The Random Forest model typically achieves superior performance (~94% accuracy) compared to the Naive Bayes baseline (~83% accuracy).

## Libraries Used
- `pandas`, `numpy` for data handling
- `matplotlib`, `seaborn` for data visualization
- `scikit-learn` for machine learning models and metrics
