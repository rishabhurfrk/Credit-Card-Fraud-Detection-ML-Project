# Credit Card Fraud Detection

This repository contains a machine learning project aimed at detecting fraudulent transactions from credit card data using various classification techniques.

## Project Overview

Fraud detection is a key challenge in the financial industry, where companies need to detect and prevent fraudulent activities to protect customers and reduce losses. This project builds a machine learning model to identify potentially fraudulent transactions using a dataset of credit card transactions.

The main steps involved in this project are:
1. Data loading and exploration
2. Data preprocessing (scaling, splitting, etc.)
3. Model building using various classification algorithms
4. Model evaluation and comparison
5. Fine-tuning the best model for optimal performance

## Dataset

The dataset used for this project is highly imbalanced, with only a small percentage of transactions being fraudulent. It is publicly available on [Kaggle](https://www.kaggle.com/mlg-ulb/creditcardfraud) and contains the following:
- **Time:** Seconds elapsed between this transaction and the first transaction in the dataset.
- **V1-V28:** Principal components obtained with PCA (features).
- **Amount:** Transaction amount.
- **Class:** 1 for fraudulent transactions, 0 for non-fraudulent transactions.

## Models Used

The following machine learning algorithms have been explored in this project:
- Logistic Regression
- Decision Trees
- Random Forest
- Gradient Boosting
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

## Results

Model evaluation metrics used include:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score

The final model was selected based on its performance on these metrics.

## How to Use

To run this notebook:
1. Clone the repository.
2. Ensure you have installed the required dependencies:
   ```bash
   pip install -r requirements.txt
3. Open the Jupyter Notebook and run all cells to see the results and outputs.

###Usage
1. Open the notebook ML___Credit_Card_Fraud_Detection.ipynb using Jupyter or any notebook interface.
2. Follow the instructions in each section to execute and analyze the results.
3. Modify the code if you want to try different models, or tune hyperparameters for better performance.


###License
This project is licensed under the MIT License - see the LICENSE file for details.

###Acknowledgments
Kaggle for providing the dataset.
Scikit-learn for the machine learning library used.