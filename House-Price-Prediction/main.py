import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler


df = pd.read_csv('cleaned_house_price_data.csv')  


print("\n=== 1. EDA ===")
print("Basic info:")
print(df.info())

print("\nNumeric stats:")
print(df.describe(include='all'))

plt.figure(figsize=(12, 5))

# Price Distribution
plt.subplot(1, 2, 1)
sns.histplot(df['Price'], bins=30, kde=True)
plt.title('Price Distribution')

# Area vs Price Scatter Plot
plt.subplot(1, 2, 2)
sns.scatterplot(x='Area', y='Price', data=df, alpha=0.6)
plt.title('Area vs Price')
plt.tight_layout()
plt.show()


print("\n=== 2. Missing Values ===")
print("Missing values per column:")
print(df.isnull().sum())

if 'Area' in df.columns:
    df['Area'] = df['Area'].fillna(df['Area'].median())
if 'Bedrooms' in df.columns:
    df['Bedrooms'] = df['Bedrooms'].fillna(df['Bedrooms'].mode()[0])


print("\n=== 3. Outliers ===")
if 'Price' in df.columns:
    Q1 = df['Price'].quantile(0.25)
    Q3 = df['Price'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = (df['Price'] < lower_bound) | (df['Price'] > upper_bound)
    print(f"Number of outliers detected: {outliers.sum()}")
    
    df['Price'] = np.where(df['Price'] > upper_bound, upper_bound, df['Price'])
    df['Price'] = np.where(df['Price'] < lower_bound, lower_bound, df['Price'])


print("\n=== 4. Categorical Encoding ===")
if 'Location' in df.columns:
    le = LabelEncoder()
    df['Location_Encoded'] = le.fit_transform(df['Location'])

if 'Bedrooms' in df.columns:
    df = pd.get_dummies(df, columns=['Bedrooms'], prefix='Bedroom')


print("\n=== 5. Feature Scaling ===")
if 'Area' in df.columns:
    df['Area_Normalized'] = MinMaxScaler().fit_transform(df[['Area']])
if 'Price' in df.columns:
    df['Price_Standardized'] = StandardScaler().fit_transform(df[['Price']])


print("\nFinal preprocessed data:")
print(df[['Location', 'Area', 'Price', 'Area_Normalized', 'Price_Standardized']].head())

df.to_csv('cleaned_house_price_data.csv', index=False)
print("\nPreprocessing complete. Cleaned data saved to 'cleaned_house_price_data.csv'")