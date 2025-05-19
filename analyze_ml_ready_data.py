import pandas as pd
import numpy as np
import logging
import os
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
            if choice in [1, 2, 3, 4]:
                return choice
            else:
                print("Please enter a valid option (1-4)")
        except ValueError:
            print("Please enter a number between 1 and 4")

def debug_data_issues(df, X, y, X_train, X_test, y_train, y_test):
    """Debug data issues that might be causing the high model accuracy."""
    logging.info("=== DEBUGGING DATA ISSUES ===")
    
    # Check for data leakage - direct leakage from target in features
    logging.info("Checking for direct data leakage...")
    for col in X.columns:
        if 'foreclosure' in col.lower() or 'source' in col.lower():
            logging.warning(f"Potential leakage in column: {col}")
    
    # Check train/test split
    logging.info(f"Train data shape: {X_train.shape}, Test data shape: {X_test.shape}")
    logging.info(f"Train class balance: {y_train.mean():.2%} foreclosures")
    logging.info(f"Test class balance: {y_test.mean():.2%} foreclosures")
    
    # Check for duplicate rows between train and test
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Count matches
    merged = pd.merge(train_df, test_df, how='inner')
    logging.info(f"Duplicate rows between train and test: {len(merged)}")
    if len(merged) > 0:
        logging.warning("Found duplicate rows between train and test sets!")
        logging.info(f"Percentage of test data duplicated in train: {len(merged)/len(test_df):.2%}")
    
    # Check correlation matrix
    logging.info("Top feature correlations with target:")
    # Convert to DataFrame for ease of manipulation
    X_y = pd.concat([X, y], axis=1)
    corrs = X_y.corr()['is_foreclosure'].sort_values(ascending=False)
    logging.info(corrs[:10].to_string())
    logging.info("Bottom feature correlations with target:")
    logging.info(corrs[-10:].to_string())
    
    # Check for missing values
    logging.info(f"Missing values in features: {X.isnull().sum().sum()}")
    
    # Check unique values for each feature to identify perfect predictors
    for col in X.columns:
        n_unique = X[col].nunique()
        logging.info(f"Feature '{col}' has {n_unique} unique values")
        
        # If few unique values, check distribution by target
        if n_unique <= 10:
            cross_tab = pd.crosstab(X[col], y)
            logging.info(f"Distribution of {col} by target:\n{cross_tab}")
    
    # Save debugging results
    if not os.path.exists('outputs/debug'):
        os.makedirs('outputs/debug')
        
    # Export samples from train and test for manual inspection
    X_train.sample(min(10, len(X_train))).to_csv('outputs/debug/train_sample.csv')
    X_test.sample(min(10, len(X_test))).to_csv('outputs/debug/test_sample.csv')
    
    # Save correlation matrix
    corrs.to_csv('outputs/debug/target_correlations.csv')
    
    logging.info("Debug information saved to outputs/debug/")

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
    
    # Use different preprocessing approaches for comparison
    # 1. Standard preprocessing (without price features to avoid leakage)
    X, y, df_processed = prepare_features(df, check_leakage=True, remove_price_features=True)
    
    # Check class balance
    foreclosure_ratio = y.mean()
    logging.info(f"Class balance: {foreclosure_ratio:.2%} foreclosures, {1-foreclosure_ratio:.2%} regular sales")
    
    # Basic EDA and correlations
    plot_data_distributions(df_processed)
    analyze_feature_correlations(df_processed)
    
    # Ensure proper train/test split without possible contamination
    # Use a deterministic but unique index to ensure consistent splits across runs
    # but avoid using potentially leaky properties like price for stratification
    
    # Create a composite stratification variable to ensure balance in key categories
    # without directly using the target (which would still result in a proper split)
    strat_var = pd.Series(0, index=y.index)
    if 'property_type_villa' in X.columns:
        strat_var += X['property_type_villa'] * 4
    if 'property_type_ejerlejlighed' in X.columns:
        strat_var += X['property_type_ejerlejlighed'] * 2
    if 'sale_year' in X.columns:
        strat_var += (X['sale_year'] > 2020).astype(int)
    
    # Split data using this balanced stratification
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strat_var
    )
    
    # Verify no overlap between train and test due to duplicated properties
    train_ids = set(X_train.index)
    test_ids = set(X_test.index) 
    overlap = train_ids.intersection(test_ids)
    if overlap:
        logging.warning(f"Found {len(overlap)} overlapping indices between train and test")
        # If there are overlaps, recreate a cleaner split
        unique_indices = sorted(list(set(X.index)))
        train_size = int(0.8 * len(unique_indices))
        train_indices = unique_indices[:train_size]
        test_indices = unique_indices[train_size:]
        X_train, X_test = X.loc[train_indices], X.loc[test_indices]
        y_train, y_test = y.loc[train_indices], y.loc[test_indices]
        logging.info(f"Created clean split - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Get user choice for model selection
    choice = get_user_choice()
    
    # Initialize model variables
    dummy_models = None
    rf_model = None
    
    # Debug mode
    if choice == 4:
        debug_data_issues(df, X, y, X_train, X_test, y_train, y_test)
        return
        
    # Train models based on user choice
    if choice in [1, 3]:  # Dummy or Both
        dummy_models = train_dummy_classifiers(X_train, X_test, y_train, y_test)
    
    if choice in [2, 3]:  # Random Forest or Both
        rf_model = train_random_forest(X_train, X_test, y_train, y_test, X)
    
    # If all models were run, create comparison
    if choice == 3:
        models_dict = {
            'Dummy (most_frequent)': dummy_models.get('most_frequent'),
            'Dummy (stratified)': dummy_models.get('stratified'),
            'Random Forest': rf_model
        }
        compare_models(models_dict, X_test, y_test)
    
    logging.info("Analysis complete. Check the output folders for results.")
    
if __name__ == "__main__":
    main() 