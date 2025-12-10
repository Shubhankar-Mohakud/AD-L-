import numpy as np
import pandas as pd
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

# Select a single feature for 2D plotting
X = df[['Avg. Area Income']]   # independent variable
y = df['Price']                # dependent variable

# Create a Linear Regression model
model = LinearRegression()

# Fit the model to the data
model.fit(X, y)

# Predict y values
y_pred = model.predict(X)

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y, y_pred)

# Calculate R^2 Score
r2 = r2_score(y, y_pred)

# Print the evaluation metrics
print(f"Mean Squared Error (MSE): {mse}")
print(f"R^2 Score: {r2}")

# Plotting the data points and the regression line
plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_pred, color='red', label='Regression line')
plt.xlabel('X (Independent variable)')
plt.ylabel('y (Dependent variable)')
plt.title('Simple Linear Regression using scikit-learn')
plt.legend()
plt.show()

