import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

data = {
    'Customer ID': ['C001', 'C002', 'C003', 'C004', 'C005','C006', 'C007', 'C008'],
    'Age': [25, 35, np.nan, 42, 28, 55, 31, 45],
    'Gender': ['Male', 'Female', 'Male', np.nan, 'Female', 'Male', 'Female', 'Male'],
    'Income': [50000, 75000, 60000, np.nan, 45000, 90000, 55000, 80000],
    'City': ['Urban', 'Rural', 'Urban', 'Urban', np.nan, 'Rural', 'Urban', 'Rural'],
    'Subscription Status': ['Subscribed', 'Not Subscribed', 'Subscribed', 'Subscribed', 
                            'Not Subscribed', np.nan, 'Subscribed', 'Not Subscribed']
}

df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)
print("\n" + "="*65 + "\n")


print("MISSING VALUES CHECK")
print("-" * 40)
print("Missing values per column:")
print(df.isna().sum())
print(f"\nTotal missing values: {df.isna().sum().sum()}")
print("\n")

#II. Handle Missing Vlaues
"""
## For numerical Values
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Income'].fillna(df['Income'].median(), inplace=True)

## For Categorical Values
df['Gender'].fillna(df['Gender'].mode(), inplace=True)
df['City'].fillna(df['City'].mode(), inplace=True)
df['Subscription Status'].fillna(df['Subscription Status'].mode()[0], inplace=True)

"""
fill_values = {
    'Age': df['Age'].median(),
    'Income': df['Income'].median(),
    'Gender': df['Gender'].mode()[0],
    'City': df['City'].mode()[0],
    'Subscription Status': df['Subscription Status'].mode()[0]
}

df.fillna(fill_values, inplace=True)

print("\nValues Filled :")
print(fill_values)

print("After handling missing values:")
print(df)
print(f"\nRemaining missing values: {df.isna().sum().sum()}")
print("\n" + "="*65 + "\n")

## Label Encoding
le_gender = LabelEncoder()
le_city = LabelEncoder()
le_subscription = LabelEncoder()

df['Gender_Encoded'] = le_gender.fit_transform(df['Gender'])
df['City_Encoded'] = le_city.fit_transform(df['City'])
df['Subscription_Encoded'] = le_subscription.fit_transform(df['Subscription Status'])

print("Label Encoding applied:")
print(f"Gender mapping: {dict(zip(le_gender.classes_, le_gender.transform(le_gender.classes_)))}")
print(f"City mapping: {dict(zip(le_city.classes_, le_city.transform(le_city.classes_)))}")
print(f"Subscription mapping: {dict(zip(le_subscription.classes_, le_subscription.transform(le_subscription.classes_)))}")
print("\n")

# III. Feature Scaling
print("III. FEATURE SCALING (MinMaxScaler)")
print("-" * 40)

# Create a copy for scaling
df_scaled = df.copy()

# Initialize MinMaxScaler
scaler = MinMaxScaler()

# Scale Age and Income
numerical_cols = ['Age', 'Income']
df_scaled[['Age_Scaled', 'Income_Scaled']] = scaler.fit_transform(df[numerical_cols])

print("Before scaling:")
print(df[['Customer ID', 'Age', 'Income']])
print("\nAfter scaling (0-1 range):")
print(df_scaled[['Customer ID', 'Age_Scaled', 'Income_Scaled']])
print("\n" + "="*65 + "\n")

# Final DataFrame with all preprocessing
print("FINAL PREPROCESSED DATAFRAME:")
print("-" * 40)
final_df = df_scaled[['Customer ID', 'Age_Scaled', 'Income_Scaled', 
                      'Gender_Encoded', 'City_Encoded', 'Subscription_Encoded']]
print(final_df)
print("\n")

# Summary statistics
print("Summary Statistics:")
print(final_df.describe())