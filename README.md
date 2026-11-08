# Breast Cancer Classification Using Machine Learning

## 1. Problem Statement

The objective of this project is to implement and compare five
supervised machine-learning classification algorithms on the Breast
Cancer Wisconsin (Diagnostic) dataset. The models classify a tumour
diagnosis as **Benign (B)** or **Malignant (M)** using numerical
features computed from digitized images of fine-needle aspirates (FNA)
of breast masses.

The five classifiers are Logistic Regression, Decision Tree, K-Nearest
Neighbors (KNN), Gaussian Naive Bayes, and Support Vector Machine (SVM).
Their performance is compared using Accuracy, Area Under the ROC Curve
(AUC), Precision, Recall, F1-score, and Matthews Correlation Coefficient
(MCC).

This project is for educational and machine-learning evaluation purposes
only. It is **not a medical diagnostic tool**.

## 2. Dataset Description

-   **Dataset:** Breast Cancer Wisconsin (Diagnostic)
-   **Source:** UCI Machine Learning Repository ---
    https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
-   **Number of instances:** 569
-   **Number of input features:** 30 numerical features
-   **Target column:** `Diagnosis`
-   **Target classes:** Benign (`B`) and Malignant (`M`)
-   **Class distribution:** 357 benign cases and 212 malignant cases
-   **Preprocessing:** The ID column is excluded from model inputs. The
    target is encoded as `B = 0` and `M = 1`, making malignant the
    positive class. The data is split into training and held-out test
    sets using a stratified split. The training data is oversampled to
    address class imbalance; the held-out test set is not oversampled.

The 30 input features describe characteristics such as radius, texture,
perimeter, area, smoothness, compactness, concavity, concave points,
symmetry, and fractal dimension. Measurements are provided as mean,
standard error (`se`), and worst-value features.

## 3. GitHub Repository Link

**GitHub Repository:** https://github.com/Lasya-2002/ML-Assignment_2

The repository includes the Streamlit application, dependency list,
dataset test file, source notebook, saved model artifacts, and this
README file.

**Live Streamlit Application:**
https://ml-assignment2-2026aj05335.streamlit.app/ ---

## 4. Models Used and Evaluation

All five models were evaluated on the same held-out test set. The
positive class for Precision, Recall, F1-score, and ROC-AUC
interpretation is **Malignant (M)**. The values below are taken from the
executed notebook's model-comparison output and are rounded to four
decimal places.

### 4.1 Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.9737 | 0.9944 | 0.9756 | 0.9524 | 0.9639 | 0.9433 |
| Decision Tree | 0.9211 | 0.9127 | 0.9024 | 0.8810 | 0.8916 | 0.8297 |
| KNN | 0.9561 | 0.9836 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Naive Bayes (Gaussian) | 0.9298 | 0.9947 | 0.9722 | 0.8333 | 0.8974 | 0.8504 |
| Support Vector Machine (SVM) | 0.9825 | 0.9947 | 0.9762 | 0.9762 | 0.9762 | 0.9623 |


### 4.2 Observations About Model Performance

| ML Model Name | Observation About Model Performance |
| :--- | :--- |
| **Logistic Regression** | Achieved high accuracy (0.9737) and AUC (0.9944). Its precision (0.9756) and recall (0.9524) indicate that it identifies malignant cases well while maintaining a strong balance between false-positive and false-negative predictions. Its MCC of 0.9433 also indicates strong overall classification agreement. |
| **Decision Tree** | Recorded an accuracy of 0.9211 and AUC of 0.9127, the lowest values among the five models in this comparison. Its precision (0.9024), recall (0.8810), F1-score (0.8916), and MCC (0.8297) are also lower than those of the other models. This suggests comparatively weaker generalization on this held-out test set. |
| **KNN** | Achieved an accuracy of 0.9561 and AUC of 0.9836. Its precision is high (0.9744), but recall (0.9048) is lower than that of Logistic Regression and SVM. This means its malignant predictions are usually correct, but it misses more malignant cases than the top-performing model in this run. |
| **Naive Bayes (Gaussian)** | Produced a high AUC of 0.9947 and precision of 0.9722, but its recall (0.8333) is the lowest of the five models. Although its ranking ability as reflected by AUC is strong, its selected classification threshold results in more missed malignant cases. Accuracy is 0.9298 and MCC is 0.8504. |
| **Support Vector Machine (SVM)** | Achieved the highest accuracy (0.9825), tied-highest AUC (0.9947), highest precision (0.9762), highest recall (0.9762), highest F1-score (0.9762), and highest MCC (0.9623) in this run. Its strong results across multiple metrics indicate the best overall performance on the held-out test set among the models evaluated. |


### 4.3 Overall Winner for This Dataset

**Overall winner: Support Vector Machine (SVM).**

SVM is the strongest overall model in this experiment because it
achieved the highest accuracy, precision, recall, F1-score, and MCC,
while also sharing the highest AUC with Gaussian Naive Bayes. In
particular, its recall of 0.9762 indicates that it identified a high
proportion of malignant cases in the held-out test set.

The conclusion applies to this dataset, split, preprocessing, and model
configuration. Different splits or hyperparameters may produce different
results. For a medical context, false negatives are especially
important, so recall and the confusion matrix should be considered
alongside accuracy; these models should not be used to make real
clinical decisions.

## 5. How to Run the Streamlit Application

1.  Clone or download the GitHub repository.

2.  Install the dependencies:

    ``` bash
    pip install -r requirements.txt
    ```

3.  Ensure the saved model files are present in the `model/` directory.

4.  Start the app:

    ``` bash
    streamlit run app.py
    ```

5.  Upload a labelled test CSV containing the required predictor columns
    and the `Diagnosis` target column. Select a model to view its
    evaluation metrics and confusion matrix/classification report. The
    application loads saved models; it does not retrain them on the
    uploaded test file.

## 6. Repository Contents

``` text
project_folder/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── Breast_Cancer_Classification_Assignment(1).ipynb
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── gaussian_nb.joblib
    ├── svm.joblib
    └── metadata.json
```
