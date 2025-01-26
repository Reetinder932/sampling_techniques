import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import requests

url = "https://github.com/AnjulaMehto/Sampling_Assignment/raw/main/Creditcard_data.csv"
data = pd.read_csv(url)

X = data.drop('Class', axis=1) 
y = data['Class']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def balance_data(sampling_type, X_train, y_train):
    if sampling_type == 'oversampling':
        sampler = RandomOverSampler(random_state=42)
    elif sampling_type == 'undersampling':
        sampler = RandomUnderSampler(random_state=42)
    elif sampling_type == 'smote':
        sampler = SMOTE(random_state=42)
    X_res, y_res = sampler.fit_resample(X_train, y_train)
    return X_res, y_res

models = {
    "M1": LogisticRegression(),
    "M2": RandomForestClassifier(),
    "M3": DecisionTreeClassifier(),
    "M4": SVC(),
    "M5": KNeighborsClassifier()
}

sampling_techniques = ['oversampling', 'undersampling', 'smote']  
results = {}

for model_name, model in models.items():
    model_results = []
    for technique in sampling_techniques:
        X_res, y_res = balance_data(technique, X_train, y_train)
        model.fit(X_res, y_res)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        model_results.append(accuracy)
    results[model_name] = model_results

for model_name, accuracies in results.items():
    print(f"Results for {model_name}:")
    for i, technique in enumerate(sampling_techniques):
        print(f"  {technique}: {accuracies[i]:.2f}")

best_technique = {model_name: sampling_techniques[np.argmax(accuracies)] for model_name, accuracies in results.items()}
print("Best Sampling Technique for Each Model:", best_technique)
