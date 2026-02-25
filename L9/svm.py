# SVM on Iris Dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, ConfusionMatrixDisplay)

# ─────────────────────────────────────────────
# 1. Load & Explore Data
# ─────────────────────────────────────────────
filePath = Path(r"C:\Users\KIIT0001\Desktop\Coding\AD-L-\L9\iris.csv")
df = pd.read_csv(filePath)
print("Shape:", df.shape)
print("\nFirst 5 rows:\n", df.head())
print("\nClass distribution:\n", df.iloc[:, -1].value_counts())

# ─────────────────────────────────────────────
# 2. Preprocess
# ─────────────────────────────────────────────
X = df.iloc[:, :-1].values          # features
y = df.iloc[:, -1].values           # target (species)

# Encode string labels if necessary
le = LabelEncoder()
y = le.fit_transform(y)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train / Test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain size: {X_train.shape[0]}  |  Test size: {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 3. Train SVM with different kernels
# ─────────────────────────────────────────────
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
kernel_scores = {}

print("\n── Kernel Comparison ──")
for kernel in kernels:
    svm = SVC(kernel=kernel, random_state=42)
    cv_scores = cross_val_score(svm, X_scaled, y, cv=5)
    kernel_scores[kernel] = cv_scores.mean()
    print(f"  {kernel:8s}  CV Accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

best_kernel = max(kernel_scores, key=kernel_scores.get)
print(f"\nBest kernel: {best_kernel}  ({kernel_scores[best_kernel]:.4f})")

# ─────────────────────────────────────────────
# 4. Hyperparameter Tuning (GridSearchCV)
# ─────────────────────────────────────────────
param_grid = {
    'C':     [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1],
    'kernel': [best_kernel]
}

grid_search = GridSearchCV(SVC(random_state=42), param_grid,
                           cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("\n── GridSearchCV Results ──")
print("Best params :", grid_search.best_params_)
print("Best CV acc :", round(grid_search.best_score_, 4))

# ─────────────────────────────────────────────
# 5. Evaluate Best Model on Test Set
# ─────────────────────────────────────────────
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

print("\n── Test Set Evaluation ──")
print(f"Accuracy : {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# ─────────────────────────────────────────────
# 6. Visualisations
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("SVM on Iris Dataset", fontsize=14, fontweight='bold')

# — Confusion Matrix —
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
disp.plot(ax=axes[0], colorbar=False, cmap='Blues')
axes[0].set_title("Confusion Matrix")

# — Kernel Accuracy Comparison —
axes[1].bar(kernel_scores.keys(), kernel_scores.values(),
            color=['#4C72B0','#DD8452','#55A868','#C44E52'], edgecolor='white')
axes[1].set_ylim(0.8, 1.02)
axes[1].set_title("5-Fold CV Accuracy by Kernel")
axes[1].set_ylabel("Accuracy")
for i, (k, v) in enumerate(kernel_scores.items()):
    axes[1].text(i, v + 0.005, f"{v:.3f}", ha='center', fontsize=10)

plt.tight_layout()
plt.savefig("svm_iris_results.png", dpi=150, bbox_inches='tight')
plt.show()
print("\nPlot saved as svm_iris_results.png")