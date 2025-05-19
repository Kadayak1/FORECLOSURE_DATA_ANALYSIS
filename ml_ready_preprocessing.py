import pandas as pd
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(file_path='data/combined_property_data.csv'):
    """Load the combined property data CSV."""
    logging.info(f"Loading data from {file_path}")
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded data shape: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def clean_data(df):
    """Clean and normalize the dataset for ML purposes."""
    logging.info("Cleaning and normalizing data...")
    
    # Convert postal code to string
    if 'postal_code' in df.columns:
        df['postal_code'] = df['postal_code'].astype(str)
    
    # Replace 'Ikke oplyst' and empty strings with NaN
    df = df.replace(['Ikke oplyst', ''], np.nan)
    
    # Convert numeric columns to numeric, coercing errors to NaN
    numeric_cols = ['price', 'living_area', 'rooms']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Standardize categorical columns (lowercase, strip whitespace)
    categorical_cols = ['property_type', 'sale_type', 'heating_type', 'wall_material', 'roof_type']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype(str).str.lower().str.strip()
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Log missing values
    missing = df.isnull().sum()
    logging.info(f"Missing values after cleaning:\n{missing[missing > 0]}")
    
    return df

def get_clean_data():
    """Convenience function to load and clean data in one step."""
    df = load_data()
    return clean_data(df)

# Only run if executed directly (not when imported)
if __name__ == "__main__":
    df = get_clean_data()
    print("Data loaded and cleaned. Available in DataFrame 'df'.") 