# Machine Learning Assignment 2 – Heart Disease Classification

**Programme:** M.Tech. Artificial Intelligence & Machine Learning  
**Course:** Machine Learning  
**Dataset:** UCI Heart Disease  
**Problem Type:** Binary Classification

## 1. Problem Statement

The objective of this project is to implement and compare five supervised machine learning classification algorithms for predicting the presence or absence of heart disease using patient clinical attributes.

The five classification models are evaluated on the same dataset using Accuracy, Area Under the ROC Curve (AUC), Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

An interactive Streamlit application is developed to allow users to upload a test dataset, select a trained classification model, and view its evaluation metrics, confusion matrix, and classification report.

## 2. Dataset Description

**Dataset Name:** UCI Heart Disease Dataset  
**Source:** UCI Machine Learning Repository  
**Dataset URL:** https://archive.ics.uci.edu/dataset/45/heart+disease

The dataset combines records from four sources:

| Source | Number of Records |
|---|---:|
| Cleveland | 303 |
| Hungary | 294 |
| Switzerland | 123 |
| VA Long Beach | 200 |
| **Total** | **920** |

**Dataset characteristics:**

- Total instances: 920
- Input features: 13
- Target variable: `target`
- Classification type: Binary
- Training dataset: 736 records (80%)
- Testing dataset: 184 records (20%)

**Input features:** `age`, `sex`, `cp`, `trestbps`, `chol`, `fbs`, `restecg`, `thalach`, `exang`, `oldpeak`, `slope`, `ca`, `thal`.

The original target variable `num` is converted into binary classes:

- **0:** Absence of heart disease
- **1:** Presence of heart disease (original values 1–4)

Missing values are handled using imputation, categorical features are encoded, and numerical features are scaled through preprocessing pipelines. The dataset is split into training and testing subsets using stratified sampling.

## 3. GitHub Repository Link

**GitHub Repository:**  
https://github.com/amitgupta0925/heart-disease-ml-classification

The repository contains the complete Python source code, `requirements.txt`, test dataset (`test_data.csv`), five saved trained models, and Streamlit application.

## 4. Models Used and Evaluation Metrics

The following five machine learning classification algorithms were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

All models were evaluated on the same 184-record test dataset using six evaluation metrics.

### 4.1 Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8152 | 0.8944 | 0.8269 | 0.8431 | 0.8350 | 0.6253 |
| Decision Tree | 0.6848 | 0.7384 | 0.7292 | 0.6863 | 0.7071 | 0.3674 |
| KNN | 0.8043 | 0.8758 | 0.8000 | 0.8627 | 0.8302 | 0.6026 |
| Gaussian Naive Bayes | 0.8043 | 0.8702 | **0.8511** | 0.7843 | 0.8163 | 0.6101 |
| Support Vector Machine | **0.8261** | **0.8984** | 0.8070 | **0.9020** | **0.8519** | **0.6487** |

*All results are calculated using the held-out test dataset. Higher values indicate better performance for these six metrics.*

### 4.2 Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---|---|
| **Logistic Regression** | Achieved 81.52% accuracy and 0.8944 AUC. It demonstrated strong classification performance with a balanced precision of 82.69% and recall of 84.31%. |
| **Decision Tree** | Achieved the lowest accuracy (68.48%) and MCC (0.3674). Its comparatively weaker test performance indicates that it was less effective than the other models on this dataset. |
| **KNN** | Achieved 80.43% accuracy, 86.27% recall, and an F1 score of 0.8302. It demonstrated good performance in identifying positive cases but performed slightly below SVM overall. |
| **Gaussian Naive Bayes** | Achieved the highest precision (85.11%) among the five models. However, its recall of 78.43% was lower than the other leading models, indicating that it missed a greater proportion of actual positive cases. |
| **Support Vector Machine** | Achieved the highest accuracy (82.61%), AUC (0.8984), recall (90.20%), F1 score (0.8519), and MCC (0.6487). It demonstrated the strongest overall performance across the evaluation metrics. |
| **Overall Winner** | **Support Vector Machine (SVM)** is selected as the best-performing model because it achieved the highest scores in five of the six evaluation metrics, particularly recall, F1 score, and MCC. |

### 4.3 Overall Conclusion

The experimental results show that Support Vector Machine provides the strongest overall classification performance for the selected UCI Heart Disease dataset.

SVM achieved an accuracy of **82.61%**, AUC of **0.8984**, recall of **90.20%**, F1 score of **0.8519**, and MCC of **0.6487**.

Although Gaussian Naive Bayes achieved the highest precision, SVM provided a stronger overall balance of performance metrics, particularly recall and F1 score.

Therefore, **Support Vector Machine is the overall winning classification model for this experiment**.

These findings are based on the held-out test dataset and do not establish clinical diagnostic suitability.

## 5. Live Streamlit Application

**Deployed Application:**  
https://heart-disease-ml-bits-assignment2.streamlit.app/

The Streamlit application provides:

- Upload of a CSV test dataset containing input features and the target column.
- Selection among all five trained classification models.
- Display of Accuracy, AUC, Precision, Recall, F1 Score, and MCC.
- Confusion matrix visualization.
- Classification report.

The application loads previously trained models and evaluates the uploaded test data without retraining the models.