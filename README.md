# Credit Card Fraud Detection

Build econometric and machine learning models to predict rare event credit card fraud from transaction data.



# Project overview


This project develops and compares machine learning models for detecting fraudulent credit card transactions. The analysis demonstrates an end-to-end classification workflow, including data exploration, feature selection, model training, hyperparameter optimization, and performance evaluation using the Precision–Recall Area Under the Curve (PR-AUC).



# Problem Statement


Credit card fraud can cause financial losses for both customers and financial institutions, making timely and accurate fraud detection essential. This project uses transaction level data to  predict if they are fraudulent. Logistic regression, random forest, and XGBoost models are trained and compared, with PR-AUC used as the primary evaluation metric given the very small proportion of fraudulent transactions.




# Dataset


This project uses the **Credit Card Fraud Detection** dataset available through [OpenML](https://www.openml.org/search?type=data&status=active&id=42175). The dataset contains **284,807 credit card transactions** made by European cardholders over a two-day period, including **492 fraudulent transactions**.

The target variable is `Class`:

- `0`: Legitimate transaction
- `1`: Fraudulent transaction

The feature set consists of the following variables:

- `V1`–`V28`: Principal components generated through a PCA transformation. The original feature names and values are not available due to confidentiality by the source author.
- `Time`: Number of seconds elapsed between each transaction and the first transaction in the dataset.
- `Amount`: Monetary value of the transaction.

The dataset is highly imbalanced, with fraudulent transactions representing approximately **0.172%** of all observations. This imbalance was considered during model evaluation, with PR-AUC used as the primary performance metric.



# Tools and Technologies

- Python
- pandas
- NumPy
- statsmodels
- Matplotlib
- Seaborn
- Optuna
- scikit-learn
- Jupyter Notebook


## Project Structure


```text
fraud-detection/
├── code/
│   ├── utils/ (contains functions used in notebooks)
│   │   ├── __init__.py
│   │   ├── describe.py
│   │   ├── model.py
│   │   └── results.py
│   ├── 00 get data.ipynb
│   ├── 01 descriptive.ipynb
│   ├── 02a model logi.ipynb
│   ├── 02b model xgb.ipynb
│   ├── 02c model rf.ipynb
│   ├── 03 results.ipynb
│   └── requirements.txt/
├── data/ (00 notebook pulls data from OpenML website, data is publicly available)
├── README.md
└── .gitignore
```





# Methodology

The project followed an end-to-end econometric and machine learning workflow for highly imbalanced binary classification.

## 1. Exploratory Data Analysis

The dataset was examined to understand its structure, class distribution, and feature relationships. Pair plots grouped by the target variable (`Class`) were used to visualize differences between legitimate and fraudulent transactions. Because fraudulent transactions represent only a small proportion of the dataset, class imbalance was considered throughout the modeling and evaluation process.

## 2. Train-Test Split

The data was divided into training and testing sets using a 60/40 split. Stratified sampling was used to preserve the proportion of legitimate and fraudulent transactions in both sets.

## 3. Feature Selection

For the logistic regression model, feature selection incorporated statistical significance and correlation analysis to consider assumptions of logistic regression.

The features were already transformed into principal components through PCA. Therefore, no extensive additional feature transformations were required.

## 4. Model Development

Three following classification models were trained and compared with out of sample data.

- Logistic
- Logistic with polynomials
- Random forest
- XGBoost

## 5. Hyperparameter Optimization

Optuna was used to optimize the hyperparameters of XGB and Randm Forests models. The optimization objective was to maximize PR-AUC, which is considered appropriate given the high imbalance of fraud in the data.


## 6. Model Evaluation

Model performance was evaluated on both the training and testing sets using Precision–Recall Area Under the Curve (PR-AUC). The test-set PR-AUC was used as the primary basis for comparing model performance and assessing generalization to unseen transactions.




# Key Findings


- The dataset was highly imbalanced, with fraud transactions representing only approximately 0.172% of all transactions. As a result, PR-AUC was used as the primary evaluation metric.
- XGBoost achieved the best performance on the test set, with a PR-AUC of **0.841**.
- Random forest achieved a training PR-AUC of **0.918**, but its test PR-AUC decreased to **0.779**, indicating greater overfitting.
- Logistic regression achieved a test PR-AUC of **0.672**, performing worse than both tree-based models. However, this improves to **0.760** when controlling for non linear effects.
- The test-set results indicate that XGBoost generalized best to unseen transactions among the models evaluated.
- Optimizing model hyperparameters using Optuna improved the focus on identifying fraudulent transactions rather than simply maximizing overall classification accuracy.



# 12. Limitations

- Data is already provided with PCA transformation. This limits exploring alternative transformations.
- Given PCA transformation is already present, some leakage is likely to test data.
- Similarly, while optimizing hyperparameters, PCA transformation leaks information to the holdout during k-fold validation.
- Data from 2 days is unlikely to generalize well over a calendar year, month, or even week. Fraud activity tends to vary across these periods.
- Rare event makes it difficult to sample a similar training and test datasets. Even at a high hold out rate, there were some differences in distribution of attributes across samples.
