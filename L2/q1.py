import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
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

target = "Price"
features = df.columns.drop(target)

results = []

for col in features:
    X = df[[col]]          # single feature
    y = df[target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Feature": col,
        "MSE": mse,
        "R2": r2
    })

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Find column with lowest MSE
best_feature = results_df.loc[results_df["MSE"].idxmin()]

print("Results for all single-feature models:\n")
print(results_df)

print("\nBest single predictor based on lowest MSE:")
print(best_feature)

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

# Plotting the data points and the regression line
plt.figure(figsize=(6, 4))
plt.scatter(X, y, color='blue', label='Data points')
plt.plot(X, y_pred, color='red', label='Regression line')
plt.xlabel('X (Avg. Area Income)')
plt.ylabel('y (Price)')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()
