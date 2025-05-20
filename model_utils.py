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
    
    # Create binary target based on sale_type (1 for foreclosure auction, 0 for not)
    # This is more direct than using the 'source' column
    df_fe['is_foreclosure'] = df_fe['Sale Type'].apply(lambda x: 1 if x.lower() == 'tvangsauktion' else 0)
    
    # Extract features from sale_date
    df_fe['sale_date'] = pd.to_datetime(df_fe['Sale Date'], format='%d-%m-%Y', errors='coerce')
    df_fe['sale_year'] = df_fe['sale_date'].dt.year
    df_fe['sale_month'] = df_fe['sale_date'].dt.month
    df_fe['sale_quarter'] = df_fe['sale_date'].dt.quarter
    
    # Create day of week feature (0 = Monday, 6 = Sunday)
    df_fe['sale_day_of_week'] = df_fe['sale_date'].dt.dayofweek
    
    # Create is_weekend feature
    df_fe['is_weekend'] = df_fe['sale_day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Create postal_region feature (first two digits of postal code)
    df_fe['postal_region'] = df_fe['Postal_Code'].astype(str).str[:2]
    df_fe.loc[df_fe['postal_region'] == 'na', 'postal_region'] = 'na'
    
    # Bin cities with low counts to reduce dimensionality
    city_counts = df_fe['City'].value_counts()
    cities_to_keep = city_counts[city_counts >= 10].index
    df_fe['city_binned'] = df_fe['City'].apply(lambda x: x if x in cities_to_keep else 'other')
    
    # Add season feature
    df_fe['season'] = df_fe['sale_month'].apply(
        lambda month: 'Winter' if month in [12, 1, 2] 
        else 'Spring' if month in [3, 4, 5] 
        else 'Summer' if month in [6, 7, 8] 
        else 'Fall'
    )
    
    # Create interaction features that might be predictive
    df_fe['property_season'] = df_fe['Property_Type'] + '_' + df_fe['season']
    
    # Handle missing values more intelligently
    # Fill missing floor_count with median by property_type if it exists
    if 'floor_count' in df_fe.columns:
        floor_medians = df_fe.groupby('Property_Type')['floor_count'].median()
        for prop_type in df_fe['Property_Type'].unique():
            mask = (df_fe['Property_Type'] == prop_type) & (df_fe['floor_count'].isna())
            df_fe.loc[mask, 'floor_count'] = floor_medians.get(prop_type, df_fe['floor_count'].median())
    
    # Fill missing rooms with median by property_type
    room_medians = df_fe.groupby('Property_Type')['Rooms'].median()
    for prop_type in df_fe['Property_Type'].unique():
        mask = (df_fe['Property_Type'] == prop_type) & (df_fe['Rooms'].isna())
        df_fe.loc[mask, 'Rooms'] = room_medians.get(prop_type, df_fe['Rooms'].median())
    
    # Fill missing living_area with median by property_type
    area_medians = df_fe.groupby('Property_Type')['Living_Area'].median()
    for prop_type in df_fe['Property_Type'].unique():
        mask = (df_fe['Property_Type'] == prop_type) & (df_fe['Living_Area'].isna())
        df_fe.loc[mask, 'Living_Area'] = area_medians.get(prop_type, df_fe['Living_Area'].median())
    
    # NEW FEATURE 3: Create rolling statistics
    # We need to sort by date for rolling statistics to make sense
    df_fe = df_fe.sort_values('sale_date')
    
    # Group by postal_region and calculate rolling averages
    # First, create a groupby object
    if len(df_fe) > 0 and not df_fe['postal_region'].isna().all():
        postal_groups = df_fe.groupby('postal_region')
        
        # Calculate 3-month rolling average of prices
        rolling_results = []
        
        for name, group in postal_groups:
            if 'Price' in group.columns and len(group) >= 3:
                group = group.sort_values('sale_date')
                # Calculate 3-month rolling mean of prices
                group['rolling_3m_price_mean'] = group['Price'].rolling(window=3, min_periods=1).mean()
                # Calculate year-on-year percentage changes if we have enough data
                if len(group) > 12:
                    group['price_yoy_change'] = group['Price'].pct_change(periods=12)
                else:
                    group['price_yoy_change'] = np.nan
                rolling_results.append(group)
        
        if rolling_results:
            df_fe = pd.concat(rolling_results)
    
    # Fill NaNs in new columns with appropriate values
    if 'rolling_3m_price_mean' in df_fe.columns:
        df_fe['rolling_3m_price_mean'] = df_fe['rolling_3m_price_mean'].fillna(df_fe['Price'])
    
    if 'price_yoy_change' in df_fe.columns:
        df_fe['price_yoy_change'] = df_fe['price_yoy_change'].fillna(0)
    
    return df_fe

def prepare_features(df, check_leakage=True, remove_price_features=True, feature_set='no_temporal'):
    """Prepare features for ML: encode categoricals and handle missing values.
    
    Parameters:
    -----------
    df : DataFrame
        Raw data
    check_leakage : bool
        Whether to check for and remove potential target leakage
    remove_price_features : bool
        Whether to remove price features that may create artifactual correlations
    feature_set : str
        Which feature set to use: 'full', 'no_temporal', or 'no_price_temporal'
        
    Returns:
    --------
    tuple
        (X, y, df_processed) - features, target, and processed dataframe
    """
    logging.info(f"Preparing features for ML using feature set: {feature_set}...")
    
    # Engineer features
    df_ml = engineer_features(df)
    
    # Drop rows with missing target
    df_ml = df_ml.dropna(subset=['is_foreclosure'])
    logging.info(f"Shape after dropping missing target: {df_ml.shape}")
    
    # Define temporal features
    temporal_features = ['sale_year', 'sale_month', 'sale_quarter', 'sale_day_of_week', 'economic_era',
                         'season', 'is_weekend', 'property_season'] 
    
    # Define price-related features
    price_features = ['Price', 'rolling_3m_price_mean', 'price_yoy_change']
    
    # Get target
    y = df_ml['is_foreclosure']
    
    # Basic exclude columns (always excluded)
    basic_exclude = ['Property ID', 'Address', 'Link', 'URL', 'is_foreclosure', 
                    'sale_date', 'Sale Date', 'Weighted_Area', 'transaction_id', 'Sale Type']
    
    # Configure feature inclusion based on feature_set parameter
    if feature_set == 'full':
        logging.info("Using ALL features (including temporal and price features)")
        exclude_cols = basic_exclude.copy()
        # Keep price features unless explicitly told to remove them
        if remove_price_features:
            exclude_cols.extend(price_features)
        
        # Separate numeric and categorical features
        numeric_features = ['Living_Area', 'Rooms', 'floor_count', 'is_weekend', 
                          'sale_year', 'sale_month', 'sale_quarter', 'sale_day_of_week']
        
        # Add yearly_foreclosure_rate if it exists
        if 'yearly_foreclosure_rate' in df_ml.columns:
            numeric_features.append('yearly_foreclosure_rate')
            
    elif feature_set == 'no_temporal':
        logging.info("Excluding temporal features")
        exclude_cols = basic_exclude + temporal_features
        
        # Separate numeric and categorical features
        numeric_features = ['Living_Area', 'Rooms', 'floor_count']
        
        # Keep price features unless explicitly told to remove them
        if not remove_price_features:
            numeric_features.extend([f for f in price_features if f in df_ml.columns])
            
    elif feature_set == 'no_price_temporal':
        logging.info("Excluding both price and temporal features")
        exclude_cols = basic_exclude + temporal_features + price_features
        
        # Separate numeric and categorical features - only property characteristics
        numeric_features = ['Living_Area', 'Rooms', 'floor_count']
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")
    
    # Add rolling statistics if they exist and should be included
    if feature_set != 'no_price_temporal':
        if 'rolling_3m_price_mean' in df_ml.columns and 'rolling_3m_price_mean' not in exclude_cols:
            numeric_features.append('rolling_3m_price_mean')
        
        if 'price_yoy_change' in df_ml.columns and 'price_yoy_change' not in exclude_cols:
            numeric_features.append('price_yoy_change')
    
    # Remove features with too many missing values
    for col in numeric_features[:]:
        if col in df_ml.columns and df_ml[col].isna().mean() > 0.5:  # if more than 50% is missing
            numeric_features.remove(col)
            logging.warning(f"Removed {col} due to high percentage of missing values")
    
    # Keep only numeric features that exist in the dataframe
    numeric_features = [f for f in numeric_features if f in df_ml.columns]
    
    # Make sure all exclude columns are lowercase for case-insensitive matching
    exclude_cols_lower = [col.lower() if isinstance(col, str) else col for col in exclude_cols]
    
    # Categorical features
    categorical_features = [col for col in df_ml.columns 
                          if col not in numeric_features 
                          and col.lower() not in exclude_cols_lower
                          and col not in exclude_cols
                          and df_ml[col].dtype == 'object'
                          and df_ml[col].nunique() < 30]  # Avoid columns with too many categories
    
    # For non-full feature sets, ensure all temporal-derived features are excluded
    if feature_set != 'full':
        # Remove any categorical features that are derived from temporal features
        temporal_derived = [col for col in categorical_features 
                         if any(temp in col for temp in ['season', 'year', 'month', 'quarter', 'day', 'weekend'])]
        
        for col in temporal_derived:
            if col in categorical_features:
                categorical_features.remove(col)
                logging.info(f"Removed temporal-derived feature: {col}")

        # Specifically ensure economic_era is excluded
        if 'economic_era' in categorical_features:
            categorical_features.remove('economic_era')
            logging.info("Removed economic_era feature")
    
    logging.info(f"Final numeric columns: {numeric_features}")
    logging.info(f"Final categorical columns: {categorical_features}")
    
    # Fill missing values in numeric features with median
    for col in numeric_features:
        if col in df_ml.columns and df_ml[col].isna().any():
            df_ml[col] = df_ml[col].fillna(df_ml[col].median())
    
    # Handle categorical features with one-hot encoding
    df_encoded = pd.get_dummies(df_ml[categorical_features], drop_first=False, dummy_na=False)
    
    # Combine numeric and encoded categorical features
    X_numeric = df_ml[numeric_features].copy()
    X = pd.concat([X_numeric, df_encoded], axis=1)
    
    # Check if X has too many features and perform dimensionality reduction if needed
    if X.shape[1] > 100:
        logging.warning(f"High feature dimensionality: {X.shape[1]} features. Consider dimensionality reduction.")
        
    return X, y, df_ml

def analyze_feature_correlations(X, y, output_dir='outputs/general'):
    """Analyze correlations between features and the target variable.
    
    Parameters:
    -----------
    X : DataFrame
        Features
    y : Series
        Target variable
    output_dir : str
        Directory to save the correlation analysis
    """
    logging.info("Analyzing feature correlations...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Create a dataframe with features and target for correlation analysis
    data_corr = X.copy()
    data_corr['target'] = y
    
    # Calculate correlations with target
    target_corrs = data_corr.corr()['target'].sort_values(ascending=False)
    
    # Save top and bottom correlations to a file
    with open(f'{output_dir}/feature_correlations.txt', 'w') as f:
        f.write("Top 20 positive correlations with foreclosure status:\n")
        f.write(target_corrs.head(20).to_string())
        f.write("\n\nTop 20 negative correlations with foreclosure status:\n")
        f.write(target_corrs.tail(20).to_string())
    
    # Plot top N correlations
    top_n = 15
    plt.figure(figsize=(10, 8))
    
    # Get top positive and negative correlations (excluding target itself)
    top_pos_corrs = target_corrs[1:top_n+1]  # Skip the first which is target with itself
    top_neg_corrs = target_corrs[-(top_n):].iloc[::-1]  # Reverse to show most negative first
    
    # Plot positive correlations
    plt.subplot(2, 1, 1)
    sns.barplot(x=top_pos_corrs.values, y=top_pos_corrs.index)
    plt.title(f'Top {top_n} Positive Correlations with Foreclosure')
    plt.tight_layout()
    
    # Plot negative correlations
    plt.subplot(2, 1, 2)
    sns.barplot(x=top_neg_corrs.values, y=top_neg_corrs.index)
    plt.title(f'Top {top_n} Negative Correlations with Foreclosure')
    plt.tight_layout()
    
    plt.subplots_adjust(hspace=0.5)
    plt.savefig(f'{output_dir}/feature_correlations.png')
    
    logging.info(f"Correlation analysis saved to {output_dir}/feature_correlations.png")

def plot_data_distributions(df_processed, output_dir='outputs/general'):
    """Plot distributions of various features and save to output directory.
    
    Parameters:
    -----------
    df_processed : DataFrame
        Processed data
    output_dir : str
        Directory to save plots
    """
    logging.info("Plotting data distributions...")
    
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 1. Plot distribution of key numeric features
    numeric_features = ['Living_Area', 'Rooms', 'floor_count', 'sale_year', 'sale_month']
    
    # Add price if it exists in the dataframe
    if 'Price' in df_processed.columns:
        numeric_features.append('Price')
        
    # Add new numeric features if they exist
    if 'rolling_3m_price_mean' in df_processed.columns:
        numeric_features.append('rolling_3m_price_mean')
    
    if 'price_yoy_change' in df_processed.columns:
        numeric_features.append('price_yoy_change')
    
    # Filter list to only include features that actually exist in dataframe
    numeric_features = [f for f in numeric_features if f in df_processed.columns]
    
    # Create a figure with subplots for each numeric feature
    n_features = len(numeric_features)
    n_cols = 2
    n_rows = (n_features + 1) // 2  # Ceiling division
    
    plt.figure(figsize=(14, 4 * n_rows))
    for i, feature in enumerate(numeric_features):
        plt.subplot(n_rows, n_cols, i+1)
        
        # Plot distributions separately for each class
        sns.histplot(data=df_processed, x=feature, hue='is_foreclosure', 
                    kde=True, element='step', common_norm=False, bins=30)
        
        plt.title(f'{feature} Distribution by Class')
        plt.tight_layout()
    
    plt.savefig(f'{output_dir}/numeric_distributions.png')
    logging.info(f"Numeric feature distributions saved to {output_dir}/numeric_distributions.png")
    
    # 2. Plot class distribution
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df_processed, x='is_foreclosure')
    plt.title('Class Distribution')
    plt.xlabel('Is Foreclosure')
    plt.ylabel('Count')
    plt.xticks([0, 1], ['Regular Sale', 'Foreclosure'])
    plt.tight_layout()
    plt.savefig(f'{output_dir}/class_distribution.png')
    logging.info(f"Class distribution saved to {output_dir}/class_distribution.png")

def compare_models(models_dict, X_test=None, y_test=None, output_dir='outputs/general'):
    """Compare multiple models in a summary table and charts.
    
    Parameters:
    -----------
    models_dict : dict
        Dictionary of {model_name: model_metrics} or {model_name: model_object}
    X_test : DataFrame, optional
        Test features (only needed if models_dict contains model objects)
    y_test : Series, optional
        Test target values (only needed if models_dict contains model objects)
    output_dir : str
        Directory to save outputs
        
    Returns:
    --------
    DataFrame
        Comparison results
    """
    logging.info("Comparing all models...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comparison DataFrame
    results = []
    
    # Check if models_dict contains metrics or model objects
    first_value = next(iter(models_dict.values())) if models_dict else None
    contains_metrics = isinstance(first_value, dict)
    
    # Create a unified ROC curve plot if we have model objects
    if not contains_metrics and X_test is not None and y_test is not None:
        plt.figure(figsize=(10, 8))
        
        for name, model in models_dict.items():
            if model is not None:
                y_pred = model.predict(X_test)
                
                # Accuracy and classification report
                from sklearn.metrics import accuracy_score, classification_report
                accuracy = accuracy_score(y_test, y_pred)
                
                # Get classification report metrics
                cr = classification_report(y_test, y_pred, output_dict=True)
                precision = cr['1']['precision'] if '1' in cr else 0
                recall = cr['1']['recall'] if '1' in cr else 0
                f1 = cr['1']['f1-score'] if '1' in cr else 0
                
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
                    'F1 Score': f1,
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
    elif contains_metrics:
        # If models_dict contains metrics, use them directly
        for name, metrics in models_dict.items():
            if isinstance(metrics, dict):
                # Check if this is a metrics dictionary (like for RandomForest)
                if all(key in metrics for key in ['accuracy', 'precision', 'recall', 'f1']):
                    results.append({
                        'Model': name,
                        'Accuracy': metrics.get('accuracy', 0),
                        'Precision': metrics.get('precision', 0),
                        'Recall': metrics.get('recall', 0),
                        'F1 Score': metrics.get('f1', 0),
                        'ROC AUC': metrics.get('roc_auc', 0)
                    })
                # It might be a dict of classifier objects (like for dummy classifiers)
                elif 'most_frequent' in metrics:
                    # Extract most_frequent strategy classifier
                    classifier = metrics.get('most_frequent')
                    if classifier is not None and X_test is not None and y_test is not None:
                        # Calculate metrics for this classifier
                        y_pred = classifier.predict(X_test)
                        
                        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                        accuracy = accuracy_score(y_test, y_pred)
                        precision = precision_score(y_test, y_pred, zero_division=0)
                        recall = recall_score(y_test, y_pred, zero_division=0)
                        f1 = f1_score(y_test, y_pred, zero_division=0)
                        
                        # Try to get ROC AUC
                        try:
                            y_proba = classifier.predict_proba(X_test)[:, 1]
                            roc_auc = roc_auc_score(y_test, y_proba)
                        except:
                            roc_auc = 0.5
                        
                        results.append({
                            'Model': 'Dummy',
                            'Accuracy': accuracy,
                            'Precision': precision,
                            'Recall': recall,
                            'F1 Score': f1,
                            'ROC AUC': roc_auc
                        })
            else:
                # For other formats
                logging.warning(f"Unexpected metrics format for {name}")
    
    # Create and save comparison table
    results_df = pd.DataFrame(results)
    results_df.to_csv(f'{output_dir}/model_comparison.csv', index=False)
    
    # Only create visualizations if we have results
    if not results_df.empty:
        # Convert any non-numeric values to numeric for display
        for col in results_df.columns:
            if col != 'Model':  # Skip the model name column
                results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
                
        # Format values for the table display
        display_values = []
        for row in results_df.values:
            formatted_row = []
            for i, val in enumerate(row):
                if i == 0:  # Model name column
                    formatted_row.append(str(val))
                else:
                    try:
                        formatted_row.append(f"{val:.3f}")
                    except (ValueError, TypeError):
                        formatted_row.append(str(val))
            display_values.append(formatted_row)
        
        # Create a visual table
        plt.figure(figsize=(12, 6))
        plt.axis('off')
        table = plt.table(
            cellText=display_values,
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
        
        # Create bar chart comparison for numeric columns
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
        
        # Debugging to print column names
        logging.info(f"Available columns in results_df: {list(results_df.columns)}")
        
        for metric in metrics:
            if metric in results_df.columns:
                logging.info(f"Creating chart for {metric}...")
                plt.figure(figsize=(12, 6))
                ax = sns.barplot(x='Model', y=metric, data=results_df)
                
                # Add value labels above each bar
                for i, v in enumerate(results_df[metric]):
                    ax.text(i, v + 0.02, f"{v:.2f}", ha='center')
                
                plt.title(f'{metric} Comparison Across Models')
                plt.xticks(rotation=45, ha='right')
                plt.ylim(0, 1.0)  # Set fixed y-axis limit to ensure bars are visible
                plt.tight_layout()
                
                # Generate file name and save
                file_name = f'{output_dir}/{metric.lower().replace(" ", "_")}_comparison.png'
                plt.savefig(file_name)
                logging.info(f"Saved comparison chart to {file_name}")
                plt.close()
    
    logging.info("Model comparison complete.")
    return results_df 