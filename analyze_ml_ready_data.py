import pandas as pd
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Import the preprocessing module
from ml_ready_preprocessing import get_clean_data

# Import modular components
from model_utils import (
    create_folders, prepare_features, analyze_feature_correlations,
    plot_data_distributions, compare_models
)
from dummy_classifier import train_dummy_classifiers
from random_forest_classifier import train_random_forest

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_user_choice():
    """Get user choice of models to run through interactive prompt."""
    print("\n=== Foreclosure Analysis Model Selection ===")
    print("1: Run Enhanced Dummy Classifiers")
    print("2: Run Random Forest Classifier")
    print("3: Run both classifiers")
    print("4: Debug data preprocessing")
    
    while True:
        try:
            choice = int(input("\nEnter your choice (1-4): "))
            if 1 <= choice <= 4:
                return choice
            else:
                print("Please enter a number between 1 and 4.")
        except ValueError:
            print("Please enter a valid number.")

def debug_data_preprocessing(df):
    """Debug data preprocessing to identify potential issues."""
    logging.info("Starting data preprocessing debug...")
    
    # Check for temporal patterns
    logging.info("Checking temporal patterns...")
    df['sale_date'] = pd.to_datetime(df['sale_date'], format='%d-%m-%Y', errors='coerce')
    
    # Examine foreclosure distribution over time
    foreclosure_by_year = df.groupby([df['source'], df['sale_date'].dt.year])['property_id'].count().unstack(0)
    
    # Plot historical trend
    plt.figure(figsize=(12, 6))
    if 'foreclosure' in foreclosure_by_year.columns:
        plt.plot(foreclosure_by_year.index, foreclosure_by_year['foreclosure'], 
                label='Foreclosures', marker='o', linewidth=2)
    if 'regular' in foreclosure_by_year.columns:
        plt.plot(foreclosure_by_year.index, foreclosure_by_year['regular'], 
                label='Regular Sales', marker='x', linewidth=2)
    plt.title('Sales by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Sales')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('outputs/general/temporal_distribution.png')
    
    # Examine property type distributions
    prop_types = pd.crosstab(df['source'], df['property_type'])
    prop_pct = prop_types.div(prop_types.sum(axis=1), axis=0) * 100
    
    plt.figure(figsize=(12, 6))
    prop_pct.plot(kind='bar')
    plt.title('Property Types by Source (%)')
    plt.ylabel('Percentage')
    plt.tight_layout()
    plt.savefig('outputs/general/property_type_distribution.png')
    
    # Create debug output file
    with open('outputs/general/debug_info.txt', 'w') as f:
        f.write("Data Distribution Analysis\n")
        f.write("=========================\n\n")
        
        # Class balance
        class_balance = df['source'].value_counts(normalize=True)
        f.write(f"Class balance:\n{class_balance.to_string()}\n\n")
        
        # Property type distribution
        f.write(f"Property types by source:\n{prop_types.to_string()}\n\n")
        f.write(f"Property types percentage by source:\n{prop_pct.round(1).to_string()}\n\n")
        
        # Temporal ranges
        f.write("Date ranges:\n")
        for source in df['source'].unique():
            min_date = df[df['source'] == source]['sale_date'].min()
            max_date = df[df['source'] == source]['sale_date'].max()
            f.write(f"{source}: {min_date} to {max_date}\n")
        
        # Missing values by source
        f.write("\nMissing values by source:\n")
        for source in df['source'].unique():
            src_df = df[df['source'] == source]
            f.write(f"\n{source}:\n{src_df.isnull().sum().to_string()}\n")
    
    logging.info("Debug information saved to outputs/general/debug_info.txt")
    
    return foreclosure_by_year

def main():
    # Create output folders
    create_folders()
    
    # Load and clean data directly from the source (no intermediate CSV)
    df = get_clean_data()
    
    # Print first few rows and missing values
    logging.info("First few rows of the data:")
    logging.info(df.head())
    logging.info("Missing values summary:")
    logging.info(df.isnull().sum())
    
    # Save missing values summary to file
    with open('outputs/general/missing_values.txt', 'w') as f:
        f.write(df.isnull().sum().to_string())
    
    # Get the user's choice from the menu
    choice = get_user_choice()
    
    if choice == 4:
        # Debug data preprocessing
        debug_info = debug_data_preprocessing(df)
        logging.info("Debugging completed. Check outputs/general/ for results.")
        return
    
    # Use different preprocessing approaches for comparison
    # 1. Standard preprocessing (without price features to avoid leakage)
    X, y, df_processed = prepare_features(df, check_leakage=True, remove_price_features=True)
    
    # Log class balance
    logging.info(f"Class balance: {y.mean()*100:.2f}% foreclosures, {(1-y.mean())*100:.2f}% regular sales")
    
    # Plot feature distributions
    plot_data_distributions(df_processed)
    
    # Analyze correlations between features and target
    analyze_feature_correlations(X, y)
    
    # Train-test split with stratification to maintain class balance
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    models_performance = {}
    
    # Run model(s) based on user choice
    if choice in [1, 3]:
        # Train and evaluate dummy classifiers
        dummy_metrics = train_dummy_classifiers(X_train, X_test, y_train, y_test)
        models_performance['dummy'] = dummy_metrics
    
    if choice in [2, 3]:
        # Train and evaluate random forest
        rf_metrics = train_random_forest(X_train, X_test, y_train, y_test, X)
        models_performance['random_forest'] = rf_metrics
    
    # If running both models, compare their performance
    if choice == 3:
        compare_models(models_performance, X_test, y_test)
    
    logging.info("Analysis complete. Check the output folders for results.")

if __name__ == "__main__":
    main() 