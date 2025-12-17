import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from pathlib import Path

file_path = Path(r"C:\Users\KIIT0001\Downloads\housing_price_dataset.csv")
df = pd.read_csv(file_path)

# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())
print("\nColumn names:")
print(df.columns.tolist())

# Separate features (X) and target (y)
X = df.drop("Price", axis=1)   # all independent variables
y = df["Price"]                # dependent variable

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Multiple Linear Regression Results:")
print("-----------------------------------")
print("MSE:", mse)
print("R² Score:", round(r2, 3))

# Plot
plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred, color='blue', label='Data points')
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         linewidth=2, 
         color='red', label='Regression line')

plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Predicted vs Actual Price")
plt.grid(True)
plt.tight_layout()
plt.show()



