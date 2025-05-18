import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("Loading combined dataset...")
combined_data = pd.read_csv('combined_property_data.csv')

# Basic info
print(f"\nDataset shape: {combined_data.shape}")
print(f"\nForeclosure distribution: {combined_data['is_foreclosure'].value_counts()}")
print(f"\nForeclosure percentage: {combined_data['is_foreclosure'].mean() * 100:.2f}%")

# Convert columns to appropriate types
for col in ['Living_Area', 'Rooms', 'Price']:
    if col in combined_data.columns:
        combined_data[col] = pd.to_numeric(combined_data[col], errors='coerce')

# Check missing values
print("\nMissing values in combined dataset:")
missing = combined_data.isnull().sum()
print(missing[missing > 0])

# Basic statistics for important numerical columns
print("\nBasic statistics for key features:")
numerical_cols = ['Living_Area', 'Rooms', 'Price']
print(combined_data[numerical_cols].describe())

# Compare foreclosure vs regular properties
print("\nComparison between foreclosure and regular sales:")
for col in numerical_cols:
    regular_avg = combined_data[combined_data['is_foreclosure'] == 0][col].mean()
    foreclosure_avg = combined_data[combined_data['is_foreclosure'] == 1][col].mean()
    print(f"{col}: Regular = {regular_avg:.2f}, Foreclosure = {foreclosure_avg:.2f}, Ratio = {foreclosure_avg/regular_avg:.2f}")

# Property types distribution
if 'Property_Type' in combined_data.columns:
    print("\nProperty type distribution:")
    property_types = combined_data.groupby(['Property_Type', 'is_foreclosure']).size().unstack().fillna(0)
    print(property_types)
    
    # Calculate foreclosure rate per property type
    property_types['total'] = property_types[0] + property_types[1]
    property_types['foreclosure_rate'] = property_types[1] / property_types['total'] * 100
    print("\nForeclosure rate by property type:")
    print(property_types['foreclosure_rate'].sort_values(ascending=False)) 