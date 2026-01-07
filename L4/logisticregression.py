import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pathlib import Path

filePath = Path(r"C:\Users\KIIT0001\Desktop\Coding\AD-L-\L4\Titanic-Dataset.csv")
df = pd.read_csv(filePath)

print("Dataset Shape:", df.shape)
print(df.head())

# Select relevant columns
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]

# Handle missing values
df['Age'].fillna(df['Age'].median(), inplace=True)

# Encode categorical variable
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})

# Features & target
X = df.drop('Survived', axis=1)
y = df['Survived']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Generate values
z = np.linspace(-10, 10, 100)
sigmoid_values = sigmoid(z)

# Plot
plt.figure()
plt.plot(z, sigmoid_values)
plt.xlabel("z")
plt.ylabel("Sigmoid(z)")
plt.title("Sigmoid Function")
plt.grid(True)
plt.show()

# Linear combination (z)
z_model = np.dot(X_test, model.coef_.T) + model.intercept_

# Apply sigmoid
probabilities = sigmoid(z_model)

# Plot
plt.figure()
plt.hist(probabilities, bins=30)
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.title("Predicted Probabilities (Sigmoid Output)")
plt.show()
