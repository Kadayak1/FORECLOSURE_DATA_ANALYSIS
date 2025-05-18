import pandas as pd
import numpy as np

# Load the datasets
print("Loading datasets...")
foreclosure_data = pd.read_csv('clean_foreclosure_data.csv')
regular_data = pd.read_csv('clean_regular_data.csv')

# Add target labels
foreclosure_data['is_foreclosure'] = 1
regular_data['is_foreclosure'] = 0

# Create unique IDs for each dataset
print("Creating unique IDs...")
foreclosure_data['transaction_id'] = ['F' + str(i) for i in range(len(foreclosure_data))]
regular_data['transaction_id'] = ['R' + str(i) for i in range(len(regular_data))]

# Print column names to verify differences
print("\nForeclosure data columns:", foreclosure_data.columns.tolist())
print("\nRegular data columns:", regular_data.columns.tolist())

# Basic cleaning steps
print("\nCleaning data...")

# Convert price for regular data (has dots as thousands separators)
if 'Raw Price' in regular_data.columns:
    regular_data['Price'] = regular_data['Raw Price'].astype(str).str.replace('.', '').astype(float)
elif 'Price' in regular_data.columns:
    regular_data['Price'] = regular_data['Price'].astype(str).str.replace('.', '').astype(float)

# Create separate cleaned copies with minimal columns for successful merge
print("\nPreparing for merge...")

# Select important columns from foreclosure data
foreclosure_columns = ['transaction_id', 'is_foreclosure', 'Price', 'Living_Area', 'Rooms', 
                       'Property_Type', 'City', 'Postal_Code']
foreclosure_subset = foreclosure_data.copy()

# Select and rename columns from regular data to match foreclosure data
regular_subset = regular_data.copy()
regular_subset = regular_subset.rename(columns={
    'Property_ID': 'Property ID',
    'Property Type': 'Property_Type',
    'Living_Area': 'Living_Area',
    'Postal_Code': 'Postal_Code',
    'Rooms': 'Rooms',
    'City': 'City'
})

# Force columns to be of same type
for col in ['Living_Area', 'Rooms']:
    if col in foreclosure_subset.columns and col in regular_subset.columns:
        foreclosure_subset[col] = pd.to_numeric(foreclosure_subset[col], errors='coerce')
        regular_subset[col] = pd.to_numeric(regular_subset[col], errors='coerce')

# Extract only columns that exist in both dataframes
common_columns = []
for col in foreclosure_subset.columns:
    if col in regular_subset.columns:
        common_columns.append(col)

# Always include these columns
required_columns = ['transaction_id', 'is_foreclosure']
for col in required_columns:
    if col not in common_columns and col in foreclosure_subset.columns and col in regular_subset.columns:
        common_columns.append(col)

print(f"\nCommon columns for merging: {common_columns}")

# Select only common columns from both datasets
foreclosure_final = foreclosure_subset[common_columns].copy()
regular_final = regular_subset[common_columns].copy()

# Reset indices to ensure clean merge
foreclosure_final.reset_index(drop=True, inplace=True)
regular_final.reset_index(drop=True, inplace=True)

# Perform the merge
print("\nMerging datasets...")
try:
    combined_data = pd.concat([foreclosure_final, regular_final], ignore_index=True)
    print(f"Combined data shape: {combined_data.shape}")
    print("\nPreview of combined data:")
    print(combined_data.head())
    
    # Save the combined data
    combined_data.to_csv('combined_property_data.csv', index=False)
    print("\nSaved combined data to 'combined_property_data.csv'")
    
except Exception as e:
    print(f"\nError during merge: {e}")
    
    # Print sample rows to diagnose issue
    print("\nForeclosure sample:")
    print(foreclosure_final.head(3))
    print("\nRegular sample:")
    print(regular_final.head(3)) 