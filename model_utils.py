import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc

def create_folders():
    """Create folders for organizing outputs."""
    folders = ['outputs', 'outputs/general', 'outputs/dummy', 'outputs/random_forest']
    for folder in folders:
        if not os.path.exists(folder):
            os.makedirs(folder)
            logging.info(f"Created folder: {folder}")

def engineer_features(df):
    """Perform feature engineering on the dataset.
    
    Parameters:
    -----------
    df : DataFrame
        Raw data
        
    Returns:
    --------
    DataFrame
        Data with engineered features
    """
    logging.info("Engineering features...")
    
    # Make a copy to avoid modifying the original
    df_fe = df.copy()
    
    # Create binary target (1 for foreclosure, 0 for not)
    df_fe['is_foreclosure'] = df_fe['source'].apply(lambda x: 1 if x == 'foreclosure' else 0)
    
    # Extract features from sale_date
    df_fe['sale_date'] = pd.to_datetime(df_fe['sale_date'], format='%d-%m-%Y', errors='coerce')
    df_fe['sale_year'] = df_fe['sale_date'].dt.year
    df_fe['sale_month'] = df_fe['sale_date'].dt.month
    df_fe['sale_quarter'] = df_fe['sale_date'].dt.quarter
    
    # Calculate price per square meter
    df_fe['price_per_sqm'] = df_fe['price'] / df_fe['living_area']
    
    # Log transform price (prices are often log-normally distributed)
    df_fe['log_price'] = np.log1p(df_fe['price'])
    
    # Group cities by frequency and bin less frequent cities
    city_counts = df_fe['city'].value_counts()
    frequent_cities = city_counts[city_counts > 10].index.tolist()
    df_fe['city_binned'] = df_fe['city'].apply(lambda x: x if x in frequent_cities else 'other')
    
    # Create postal code regions (first 2 digits)
    df_fe['postal_region'] = df_fe['postal_code'].astype(str).str[:2]
    
    return df_fe

def prepare_features(df, check_leakage=True, remove_price_features=True):
    """Prepare features for ML: encode categoricals and handle missing values.
    
    Parameters:
    -----------
    df : DataFrame
        Raw data
    check_leakage : bool
        Whether to check for and remove potential target leakage
    remove_price_features : bool
        Whether to remove price features that may create artifactual correlations
        
    Returns:
    --------
    tuple
        (X, y, df_processed) - features, target, and processed dataframe
    """
    logging.info("Preparing features for ML...")
    
    # Engineer features
    df_ml = engineer_features(df)
    
    # Drop rows with missing target
    df_ml = df_ml.dropna(subset=['is_foreclosure'])
    logging.info(f"Shape after dropping missing target: {df_ml.shape}")
    
    # Select categorical and numeric features
    categorical_cols = [
        'property_type', 'heating_type', 'wall_material', 'roof_type', 
        'city_binned', 'postal_region'
    ]
    
    # Define separate price-related features that will be conditionally included
    price_features = ['price', 'price_per_sqm', 'log_price']
    
    # Define other numeric features
    other_numeric_cols = [
        'living_area', 'rooms', 'floor_count', 
        'sale_year', 'sale_month', 'sale_quarter'
    ]
    
    # Start with non-price numeric features
    numeric_cols = other_numeric_cols.copy()
    
    # Add price features only if specifically requested
    if not remove_price_features:
        numeric_cols.extend(price_features)
        logging.info("Including price-related features")
    else:
        logging.info("Excluding price-related features to avoid potential data leakage")
    
    # Check for potential leakage
    if check_leakage:
        logging.info("Checking for potential data leakage...")
        # List of columns that might contain info from the target
        potential_leakage_cols = ['source', 'sale_type', 'total_sales']
        
        # Remove columns that directly leak target information
        for col in potential_leakage_cols:
            if col in numeric_cols:
                logging.warning(f"Removing {col} from numeric features due to potential leakage")
                numeric_cols.remove(col)
            if col in categorical_cols:
                logging.warning(f"Removing {col} from categorical features due to potential leakage")
                categorical_cols.remove(col)
        
        logging.info(f"Final numeric columns: {numeric_cols}")
        logging.info(f"Final categorical columns: {categorical_cols}")
    
    # Handle missing values in numeric columns
    numeric_df = df_ml[numeric_cols].copy()
    for col in numeric_cols:
        if col in df_ml.columns:
            # Fill missing with median
            numeric_df[col] = pd.to_numeric(df_ml[col], errors='coerce')
            numeric_df[col] = numeric_df[col].fillna(numeric_df[col].median())
    
    # Encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    categorical_df = df_ml[categorical_cols].fillna('missing')
    encoded_features = encoder.fit_transform(categorical_df)
    
    # Create encoded DataFrame with the same index
    encoded_df = pd.DataFrame(
        encoded_features, 
        columns=encoder.get_feature_names_out(categorical_cols),
        index=df_ml.index
    )
    
    # Concatenate with aligned indices
    X = pd.concat([numeric_df, encoded_df], axis=1)
    y = df_ml['is_foreclosure']
    
    # Verify no leakage-related columns remain
    if check_leakage:
        leakage_words = ['foreclosure', 'source', 'sale_type']
        suspect_cols = []
        for col in X.columns:
            for word in leakage_words:
                if word in col.lower():
                    suspect_cols.append(col)
        
        if suspect_cols:
            logging.warning(f"Potential leakage columns found: {suspect_cols}")
            # Drop these columns
            X = X.drop(columns=suspect_cols)
            logging.info(f"Dropped {len(suspect_cols)} columns with potential leakage")
    
    logging.info(f"X shape: {X.shape}, y shape: {y.shape}")
    
    # Log feature names for reference
    logging.info(f"Features used: {X.columns.tolist()}")
    
    return X, y, df_ml

def analyze_feature_correlations(df_processed, output_dir='outputs/general'):
    """Analyze correlations between features and the target.
    
    Parameters:
    -----------
    df_processed : DataFrame
        Processed dataframe with features and target
    output_dir : str
        Directory to save outputs
    """
    # Create directory for correlation plots
    corr_dir = f'{output_dir}/correlations'
    os.makedirs(corr_dir, exist_ok=True)
    
    # For numeric features
    numeric_cols = ['price', 'living_area', 'rooms', 'floor_count', 'price_per_sqm', 
                   'log_price', 'total_sales', 'sale_year', 'sale_month']
    
    # Calculate point-biserial correlation (equivalent to Pearson's for binary targets)
    corrs = {}
    for col in numeric_cols:
        if col in df_processed.columns:
            corr = df_processed[col].corr(df_processed['is_foreclosure'])
            corrs[col] = corr
    
    # Sort and plot correlations
    corrs = {k: v for k, v in sorted(corrs.items(), key=lambda x: abs(x[1]), reverse=True)}
    
    plt.figure(figsize=(10, 6))
    plt.bar(corrs.keys(), corrs.values())
    plt.xticks(rotation=45, ha='right')
    plt.title('Feature Correlations with Foreclosure Status')
    plt.ylabel('Correlation Coefficient')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_correlations.png')
    plt.close()
    
    # Detailed distribution plots for top correlated features
    top_features = list(corrs.keys())[:5]  # Top 5 correlated features
    for feature in top_features:
        plt.figure(figsize=(8, 6))
        sns.boxplot(x='is_foreclosure', y=feature, data=df_processed)
        plt.title(f'{feature} by Foreclosure Status')
        plt.xlabel('Is Foreclosure (1=Yes, 0=No)')
        plt.tight_layout()
        plt.savefig(f'{corr_dir}/{feature}_boxplot.png')
        plt.close()

def plot_data_distributions(df_processed, output_dir='outputs/general'):
    """Plot distributions of key features for EDA.
    
    Parameters:
    -----------
    df_processed : DataFrame
        Processed dataframe with features and target
    output_dir : str
        Directory to save outputs
    """
    # Plot distributions
    plt.figure(figsize=(12, 10))
    plt.subplot(2, 2, 1)
    sns.histplot(data=df_processed, x='price', hue='is_foreclosure', bins=30, kde=True)
    plt.title('Price Distribution by Foreclosure Status')
    
    plt.subplot(2, 2, 2)
    sns.histplot(data=df_processed, x='living_area', hue='is_foreclosure', bins=30, kde=True)
    plt.title('Living Area Distribution by Foreclosure Status')
    
    plt.subplot(2, 2, 3)
    sns.countplot(data=df_processed, x='property_type', hue='is_foreclosure')
    plt.xticks(rotation=45)
    plt.title('Property Types by Foreclosure Status')
    
    plt.subplot(2, 2, 4)
    sns.countplot(data=df_processed, x='sale_year', hue='is_foreclosure')
    plt.title('Sale Year by Foreclosure Status')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_distributions.png')
    plt.close()
    
    # Additional plots
    # Price per square meter
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df_processed, x='price_per_sqm', hue='is_foreclosure', bins=30, kde=True)
    plt.title('Price per Square Meter by Foreclosure Status')
    plt.xlabel('Price per Square Meter')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/price_per_sqm_distribution.png')
    plt.close()
    
    # Sale month seasonality
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df_processed, x='sale_month', hue='is_foreclosure')
    plt.title('Sale Month by Foreclosure Status')
    plt.xlabel('Month')
    plt.xticks(range(12), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sale_month_distribution.png')
    plt.close()

def compare_models(models_dict, X_test, y_test, output_dir='outputs/general'):
    """Compare multiple models in a summary table and charts.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of {model_name: model_object}
    X_test : DataFrame
        Test features
    y_test : Series
        Test target values
    output_dir : str
        Directory to save outputs
        
    Returns:
    --------
    DataFrame
        Comparison results
    """
    logging.info("Comparing all models...")
    
    # Create comparison DataFrame
    results = []
    
    # Create a unified ROC curve plot
    plt.figure(figsize=(10, 8))
    
    for name, model in models_dict.items():
        if model is not None:
            y_pred = model.predict(X_test)
            
            # Accuracy and classification report
            from sklearn.metrics import accuracy_score, classification_report
            accuracy = accuracy_score(y_test, y_pred)
            
            # Get classification report metrics
            cr = classification_report(y_test, y_pred, output_dict=True)
            precision = cr['1']['precision'] 
            recall = cr['1']['recall']
            f1 = cr['1']['f1-score']
            
            # ROC AUC if available
            try:
                y_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_proba)
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')
            except:
                roc_auc = 0.5
                logging.info(f"Model {name} doesn't support predict_proba")
            
            # Add to results
            results.append({
                'Model': name,
                'Accuracy': accuracy,
                'Precision': precision,
                'Recall': recall,
                'F1': f1,
                'ROC AUC': roc_auc
            })
    
    # Finalize and save ROC curve comparison
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for All Models')
    plt.legend(loc="lower right")
    plt.savefig(f'{output_dir}/all_models_roc.png')
    plt.close()
    
    # Create and save comparison table
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{output_dir}/model_comparison.csv', index=False)
    
    # Create a visual table
    plt.figure(figsize=(12, 6))
    plt.axis('off')
    table = plt.table(
        cellText=results_df.values.round(3),
        colLabels=results_df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.savefig(f'{output_dir}/model_comparison_table.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    # Create bar chart comparison
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Model', y=metric, data=results_df)
        plt.title(f'{metric} Comparison Across Models')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/{metric.lower().replace(" ", "_")}_comparison.png')
        plt.close()
    
    logging.info("Model comparison complete.")
    return results_df 