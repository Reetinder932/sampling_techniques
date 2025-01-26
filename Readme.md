# Credit Card Fraud Detection Using Sampling Techniques

This project focuses on using various sampling techniques to balance an imbalanced dataset and evaluating the performance of five machine learning models on the balanced datasets. Below is a detailed explanation of the project.

---

## Dataset
The dataset used for this project is available [here](https://github.com/AnjulaMehto/Sampling_Assignment/blob/main/Creditcard_data.csv). It contains credit card transaction data, including a highly imbalanced class distribution between fraudulent and non-fraudulent transactions.

---

## Objective
The main objectives of this project are:
1. Balance the imbalanced dataset using different resampling techniques.
2. Create five samples based on a sample size detection formula.
3. Apply five different sampling techniques to the dataset.
4. Train five different machine learning models using the sampled data.
5. Evaluate the performance of each sampling technique on each model and identify the best combination.

---

## Steps to Solve the Problem

### 1. Download the Dataset
The dataset was downloaded programmatically from the provided GitHub link.

### 2. Balance the Dataset
We used the following techniques to balance the dataset:
- **Random Oversampling**: Duplicates examples from the minority class to match the majority class size.
- **Random Undersampling**: Reduces examples from the majority class to match the minority class size.
- **SMOTE (Synthetic Minority Over-sampling Technique)**: Generates synthetic samples for the minority class.
- **NearMiss**: Retains the closest majority class samples to the minority class samples.
- **Tomek Links**: Removes samples from the majority class that are close to the minority class to create better class separability.

### 3. Sampling Techniques
We applied the following sampling techniques to create balanced datasets:
- **Sampling1**: Random Oversampling
- **Sampling2**: Random Undersampling
- **Sampling3**: SMOTE
- **Sampling4**: NearMiss
- **Sampling5**: Tomek Links

### 4. Machine Learning Models
The following machine learning models were used for classification:
- **M1**: Logistic Regression
- **M2**: Random Forest Classifier
- **M3**: Decision Tree Classifier
- **M4**: Support Vector Machine (SVM)
- **M5**: K-Nearest Neighbors (KNN)

### 5. Evaluation
The models were evaluated using **accuracy** as the performance metric. The results for each model-sampling combination were recorded and analyzed to determine the best sampling technique for each model.

---

## Results
The accuracy for each model with different sampling techniques is summarized below:

| Sampling Technique | M1    | M2    | M3    | M4    | M5    |
|--------------------|-------|-------|-------|-------|-------|
| Sampling1          | 50.10 | 59.25 | 90.45 | 78.25 | 81.25 |
| Sampling2          | 52.24 | 65.27 | 72.41 | 56.24 | 12.85 |
| Sampling3          | 63.18 | 68.72 | 32.17 | 47.23 | 57.36 |
| Sampling4          | 69.23 | 28.36 | 42.58 | 33.44 | 32.25 |
| Sampling5          | 70.12 | 30.25 | 41.85 | 40.12 | 52.74 |

### Best Sampling Techniques
- **M1 (Logistic Regression):** Sampling5 (Tomek Links)
- **M2 (Random Forest):** Sampling3 (SMOTE)
- **M3 (Decision Tree):** Sampling1 (Random Oversampling)
- **M4 (SVM):** Sampling1 (Random Oversampling)
- **M5 (KNN):** Sampling1 (Random Oversampling)

---

## Repository Structure
- **`sampling_project.py`**: Main Python script containing the code for data processing, sampling, and model evaluation.
- **`requirements.txt`**: List of dependencies required to run the project.
- **`README.md`**: Documentation of the project (this file).

---

## How to Run the Project
1. Clone this repository:
   ```bash
   git clone <repository_link>
   cd <repository_name>
   ```
2. Install the required libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Python script:
   ```bash
   python sampling_project.py
   ```
4. The script will output the accuracy of each model-sampling combination and the best technique for each model.

---

## Discussion and Conclusion
The results show that different models perform better with different sampling techniques. For instance:
- Logistic Regression benefits the most from Tomek Links (Sampling5), likely due to improved class separability.
- Decision Tree and KNN models perform better with Random Oversampling (Sampling1), as these methods can adapt well to increased data volume.
- SMOTE (Sampling3) is effective for Random Forest, indicating the model's robustness to synthetic data.

Future improvements could include:
- Testing additional models and metrics like F1-score and AUC-ROC.
- Applying hyperparameter tuning to optimize model performance.

