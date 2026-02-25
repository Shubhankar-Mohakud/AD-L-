import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# User defined function
def confusion_matrix_elements(y_true, y_pred):
    TP = FP = TN = FN = 0
    
    for actual, predicted in zip(y_true, y_pred):
        if actual == 1 and predicted == 1:
            TP += 1
        elif actual == 0 and predicted == 1:
            FP += 1
        elif actual == 0 and predicted == 0:
            TN += 1
        elif actual == 1 and predicted == 0:
            FN += 1
            
    return TP, FP, TN, FN

def accuracy_score(y_true, y_pred):
    TP, FP, TN, FN = confusion_matrix_elements(y_true, y_pred)
    return (TP + TN) / (TP + TN + FP + FN)

def precision_score(y_true, y_pred):
    TP, FP, TN, FN = confusion_matrix_elements(y_true, y_pred)
    if TP + FP == 0:
        return 0
    return TP / (TP + FP)

def recall_score(y_true, y_pred):
    TP, FP, TN, FN = confusion_matrix_elements(y_true, y_pred)
    if TP + FN == 0:
        return 0
    return TP / (TP + FN)

def f1_score(y_true, y_pred):
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    
    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)



file_path = Path(r"C:\Users\KIIT0001\Desktop\Coding\AD-L-\L9\Titanic-Dataset.csv")
data = pd.read_csv(file_path)

selected_columns = ['Pclass', 'Survived', 'Sex', 'Age', 'Fare']
data = data[selected_columns].copy()
print(data.head())

data['Age'] = data['Age'].fillna(data['Age'].median())

le = LabelEncoder()
data['Sex'] = le.fit_transform(data['Sex'])  # female -> 0, male -> 1

X = data.drop('Survived', axis=1)
y = data['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "SVM": SVC(),
    "Random Forest": RandomForestClassifier(random_state=42)
}

print(f"{'Model':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1 Score':<10}")
print("-" * 75)

for name, model in models.items():
    # Fit and Predict
    # Use scaled data for Logistic Regression and SVM
    if name in ["Logistic Regression", "SVM"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"{name:<20} | {acc:.4f}     | {prec:.4f}    | {rec:.4f}    | {f1:.4f}")