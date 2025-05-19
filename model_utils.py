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
    df_fe['is_foreclosure'] = df_fe['sale_type'].apply(lambda x: 1 if x.lower() == 'tvangsauktion' else 0)
    
    # Extract features from sale_date
    df_fe['sale_date'] = pd.to_datetime(df_fe['sale_date'], format='%d-%m-%Y', errors='coerce')
    df_fe['sale_year'] = df_fe['sale_date'].dt.year
    df_fe['sale_month'] = df_fe['sale_date'].dt.month
    df_fe['sale_quarter'] = df_fe['sale_date'].dt.quarter
    
    # Create day of week feature (0 = Monday, 6 = Sunday)
    df_fe['sale_day_of_week'] = df_fe['sale_date'].dt.dayofweek
    
    # Create is_weekend feature
    df_fe['is_weekend'] = df_fe['sale_day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Create postal_region feature (first two digits of postal code)
    df_fe['postal_region'] = df_fe['postal_code'].astype(str).str[:2]
    df_fe.loc[df_fe['postal_region'] == 'na', 'postal_region'] = 'na'
    
    # Bin cities with low counts to reduce dimensionality
    city_counts = df_fe['city'].value_counts()
    cities_to_keep = city_counts[city_counts >= 10].index
    df_fe['city_binned'] = df_fe['city'].apply(lambda x: x if x in cities_to_keep else 'other')
    
    # Add season feature
    df_fe['season'] = df_fe['sale_month'].apply(
        lambda month: 'Winter' if month in [12, 1, 2] 
        else 'Spring' if month in [3, 4, 5] 
        else 'Summer' if month in [6, 7, 8] 
        else 'Fall'
    )
    
    # Create interaction features that might be predictive
    df_fe['property_season'] = df_fe['property_type'] + '_' + df_fe['season']
    
    # Handle missing values more intelligently
    # Fill missing floor_count with median by property_type
    floor_medians = df_fe.groupby('property_type')['floor_count'].median()
    for prop_type in df_fe['property_type'].unique():
        mask = (df_fe['property_type'] == prop_type) & (df_fe['floor_count'].isna())
        df_fe.loc[mask, 'floor_count'] = floor_medians.get(prop_type, df_fe['floor_count'].median())
    
    # Fill missing rooms with median by property_type
    room_medians = df_fe.groupby('property_type')['rooms'].median()
    for prop_type in df_fe['property_type'].unique():
        mask = (df_fe['property_type'] == prop_type) & (df_fe['rooms'].isna())
        df_fe.loc[mask, 'rooms'] = room_medians.get(prop_type, df_fe['rooms'].median())
    
    # Fill missing living_area with median by property_type
    area_medians = df_fe.groupby('property_type')['living_area'].median()
    for prop_type in df_fe['property_type'].unique():
        mask = (df_fe['property_type'] == prop_type) & (df_fe['living_area'].isna())
        df_fe.loc[mask, 'living_area'] = area_medians.get(prop_type, df_fe['living_area'].median())
    
    # NEW FEATURE 1: Yearly foreclosure rates (calculate historical rates)
    yearly_foreclosure_counts = df_fe.groupby('sale_year')['is_foreclosure'].sum()
    yearly_total_counts = df_fe.groupby('sale_year').size()
    yearly_foreclosure_rates = (yearly_foreclosure_counts / yearly_total_counts).fillna(0)
    
    # Add yearly foreclosure rate to each property
    df_fe['yearly_foreclosure_rate'] = df_fe['sale_year'].map(yearly_foreclosure_rates)
    
    # NEW FEATURE 2: Economic era binning 
    # Define economic eras
    # Pre-Financial Crisis (before 2008)
    # Financial Crisis (2008-2012)
    # Recovery Period (2013-2019)
    # COVID Period (2020-2021)
    # Post-COVID (2022 onwards)
    def assign_economic_era(year):
        if year < 2008:
            return 'pre_financial_crisis'
        elif 2008 <= year <= 2012:
            return 'financial_crisis'
        elif 2013 <= year <= 2019:
            return 'recovery_period'
        elif 2020 <= year <= 2021:
            return 'covid_period'
        else:
            return 'post_covid'
    
    df_fe['economic_era'] = df_fe['sale_year'].apply(assign_economic_era)
    
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
            if 'price' in group.columns and len(group) >= 3:
                group = group.sort_values('sale_date')
                # Calculate 3-month rolling mean of prices
                group['rolling_3m_price_mean'] = group['price'].rolling(window=3, min_periods=1).mean()
                # Calculate year-on-year percentage changes if we have enough data
                if len(group) > 12:
                    group['price_yoy_change'] = group['price'].pct_change(periods=12)
                else:
                    group['price_yoy_change'] = np.nan
                rolling_results.append(group)
        
        if rolling_results:
            df_fe = pd.concat(rolling_results)
    
    # Fill NaNs in new columns with appropriate values
    if 'rolling_3m_price_mean' in df_fe.columns:
        df_fe['rolling_3m_price_mean'] = df_fe['rolling_3m_price_mean'].fillna(df_fe['price'])
    
    if 'price_yoy_change' in df_fe.columns:
        df_fe['price_yoy_change'] = df_fe['price_yoy_change'].fillna(0)
    
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
    
    # Exclude price-related features if specified to avoid potential data leakage
    if remove_price_features:
        logging.info("Excluding price-related features to avoid potential data leakage")
        price_cols = ['price']
        df_ml = df_ml.drop(columns=price_cols, errors='ignore')
    
    # Exclude temporal features far in the future to avoid temporal leakage
    if check_leakage:
        logging.info("Checking for potential data leakage...")
        # Make sure the test set doesn't have future dates not in training set
    
    # Get target
    y = df_ml['is_foreclosure']
    
    # Separate numeric and categorical features
    numeric_features = ['living_area', 'rooms', 'floor_count', 'sale_year', 'sale_month', 
                        'sale_quarter', 'sale_day_of_week', 'is_weekend', 
                        'yearly_foreclosure_rate']  # Added yearly_foreclosure_rate
    
    # Add rolling statistics if they exist
    if 'rolling_3m_price_mean' in df_ml.columns:
        numeric_features.append('rolling_3m_price_mean')
    
    if 'price_yoy_change' in df_ml.columns:
        numeric_features.append('price_yoy_change')
    
    # Remove features with too many missing values
    for col in numeric_features[:]:
        if col in df_ml.columns and df_ml[col].isna().mean() > 0.5:  # if more than 50% is missing
            numeric_features.remove(col)
            logging.warning(f"Removed {col} due to high percentage of missing values")
    
    # Keep only numeric features that exist in the dataframe
    numeric_features = [f for f in numeric_features if f in df_ml.columns]
    
    # Categorical features (exclude certain columns)
    exclude_cols = ['property_id', 'address', 'link', 'source_site', 'url', 'is_foreclosure', 
                   'sale_date', 'source', 'weighted_area', 'postal_code', 'sale_type']  # Added sale_type to exclude cols
    if remove_price_features:
        exclude_cols.append('price')
    
    categorical_features = [col for col in df_ml.columns 
                          if col not in numeric_features 
                          and col not in exclude_cols
                          and df_ml[col].dtype == 'object'
                          and df_ml[col].nunique() < 30]  # Avoid columns with too many categories
    
    # Make sure economic_era is included in categorical features
    if 'economic_era' in df_ml.columns and 'economic_era' not in categorical_features:
        categorical_features.append('economic_era')
    
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
    
    # Log class balance
    logging.info(f"Class balance: {y.mean()*100:.2f}% foreclosures, {(1-y.mean())*100:.2f}% regular sales")
    
    # Log features used
    logging.info(f"Features used: {X.columns.tolist()}")
    
    return X, y, df_ml

def analyze_feature_correlations(X, y, output_dir='outputs/general'):
    """Analyze correlations between features and target.
    
    Parameters:
    -----------
    X : DataFrame
        Features
    y : Series
        Target variable
    output_dir : str
        Directory to save outputs
    """
    # Create a copy with target
    X_with_target = X.copy()
    X_with_target['is_foreclosure'] = y
    
    # Calculate correlation with target for all features
    corr_with_target = []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            corr = X[col].corr(y)
            corr_with_target.append((col, corr))
    
    # Sort by absolute correlation
    corr_with_target = sorted(corr_with_target, key=lambda x: abs(x[1]), reverse=True)
    
    # Prepare a dataframe for plotting
    corr_df = pd.DataFrame(corr_with_target, columns=['feature', 'correlation'])
    
    # Display top 20 correlations
    top_20 = corr_df.head(20)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    bars = plt.barh(top_20['feature'], top_20['correlation'].abs())
    
    # Color bars based on positive/negative correlation
    for i, bar in enumerate(bars):
        if top_20['correlation'].iloc[i] > 0:
            bar.set_color('green')
        else:
            bar.set_color('red')
    
    plt.title('Top 20 Feature Correlations with Foreclosure Status')
    plt.xlabel('Absolute Correlation')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_correlations.png'))
    
    # Save top correlations to CSV
    corr_df.to_csv(os.path.join(output_dir, 'feature_correlations.csv'), index=False)
    
    # Save detailed correlation analysis for later reference
    X_with_target.corr().to_csv(os.path.join(output_dir, 'full_correlation_matrix.csv'))
    
    # Also show correlation matrix for top correlated features
    top_features = top_20['feature'].tolist()
    if len(top_features) > 5:  # Ensure we have enough features for the heatmap
        plt.figure(figsize=(12, 10))
        top_corr = X_with_target[top_features + ['is_foreclosure']].corr()
        sns.heatmap(top_corr, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix for Top Features')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'top_features_correlation_matrix.png'))
    
    return corr_df

def plot_data_distributions(df_processed, output_dir='outputs/general'):
    """Plot distributions of key features to understand the data better.
    
    Parameters:
    -----------
    df_processed : DataFrame
        Processed dataframe with features and target
    output_dir : str
        Directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create distributions directory
    dist_dir = f'{output_dir}/distributions'
    os.makedirs(dist_dir, exist_ok=True)
    
    # Set plot style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # 1. Plot distribution of key numeric features
    numeric_features = ['living_area', 'rooms', 'floor_count', 'sale_year', 'sale_month']
    
    # Add price if it exists in the dataframe
    if 'price' in df_processed.columns:
        numeric_features.append('price')
        
    # Add new numeric features if they exist
    if 'yearly_foreclosure_rate' in df_processed.columns:
        numeric_features.append('yearly_foreclosure_rate')
    
    if 'rolling_3m_price_mean' in df_processed.columns:
        numeric_features.append('rolling_3m_price_mean')
        
    if 'price_yoy_change' in df_processed.columns:
        numeric_features.append('price_yoy_change')
    
    for feature in numeric_features:
        if feature in df_processed.columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df_processed, x=feature, hue='is_foreclosure', bins=30, kde=True)
            plt.title(f'Distribution of {feature} by Foreclosure Status')
            plt.xlabel(feature)
            plt.ylabel('Count')
            plt.legend(['Regular', 'Foreclosure'])
            plt.tight_layout()
            plt.savefig(f'{dist_dir}/{feature}_distribution.png')
            plt.close()
    
    # 2. Plot categorical features
    categorical_features = ['property_type', 'city_binned', 'postal_region']
    
    # Add economic era if it exists
    if 'economic_era' in df_processed.columns:
        categorical_features.append('economic_era')
    
    for feature in categorical_features:
        if feature in df_processed.columns:
            # Calculate value counts
            counts = df_processed.groupby([feature, 'is_foreclosure']).size().unstack().fillna(0)
            
            # Calculate percentage of foreclosures within each category
            if 1 in counts.columns:  # Ensure 'is_foreclosure' column exists
                counts['foreclosure_pct'] = counts[1] / (counts[0] + counts[1]) * 100
                
                # Sort by percentage for better visualization
                counts = counts.sort_values('foreclosure_pct', ascending=False)
                
                # Only keep top categories for readability
                if len(counts) > 15:
                    counts = counts.head(15)
                
                plt.figure(figsize=(12, 8))
                counts['foreclosure_pct'].plot(kind='bar')
                plt.title(f'Foreclosure Percentage by {feature}')
                plt.xlabel(feature)
                plt.ylabel('Foreclosure Percentage (%)')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(f'{dist_dir}/{feature}_foreclosure_pct.png')
                plt.close()
    
    # 3. Plot time-based features
    if 'sale_year' in df_processed.columns:
        yearly = df_processed.groupby(['sale_year', 'is_foreclosure']).size().unstack().fillna(0)
        
        plt.figure(figsize=(12, 6))
        if 1 in yearly.columns:
            plt.plot(yearly.index, yearly[1], marker='o', label='Foreclosure')
        if 0 in yearly.columns:
            plt.plot(yearly.index, yearly[0], marker='x', label='Regular')
        plt.title('Sales by Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Sales')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{dist_dir}/sales_by_year.png')
        plt.close()
        
        # Foreclosure percentage by year
        if 0 in yearly.columns and 1 in yearly.columns:
            yearly['foreclosure_pct'] = yearly[1] / (yearly[0] + yearly[1]) * 100
            
            plt.figure(figsize=(12, 6))
            plt.bar(yearly.index, yearly['foreclosure_pct'])
            plt.title('Foreclosure Percentage by Year')
            plt.xlabel('Year')
            plt.ylabel('Foreclosure Percentage (%)')
            plt.grid(True, axis='y')
            plt.savefig(f'{dist_dir}/foreclosure_percentage_by_year.png')
            plt.close()
    
    # 4. Plot yearly foreclosure rate
    if 'yearly_foreclosure_rate' in df_processed.columns and 'sale_year' in df_processed.columns:
        # Calculate average yearly foreclosure rate
        yearly_rates = df_processed.groupby('sale_year')['yearly_foreclosure_rate'].mean()
        
        plt.figure(figsize=(12, 6))
        plt.plot(yearly_rates.index, yearly_rates.values, marker='o', color='red', linewidth=2)
        plt.title('Yearly Foreclosure Rate Trend')
        plt.xlabel('Year')
        plt.ylabel('Foreclosure Rate')
        plt.grid(True)
        plt.savefig(f'{dist_dir}/yearly_foreclosure_rate_trend.png')
        plt.close()
    
    # 5. Plot foreclosure rate by economic era 
    if 'economic_era' in df_processed.columns:
        era_rates = df_processed.groupby('economic_era')['is_foreclosure'].mean() * 100
        
        # Sort eras chronologically for better visualization
        era_order = ['pre_financial_crisis', 'financial_crisis', 'recovery_period', 'covid_period', 'post_covid']
        era_rates = era_rates.reindex([e for e in era_order if e in era_rates.index])
        
        plt.figure(figsize=(12, 6))
        era_rates.plot(kind='bar', color='darkblue')
        plt.title('Foreclosure Rate by Economic Era')
        plt.xlabel('Economic Era')
        plt.ylabel('Foreclosure Rate (%)')
        plt.xticks(rotation=45, ha='right') 
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(f'{dist_dir}/foreclosure_rate_by_economic_era.png')
        plt.close()
        
    # 6. Plot rolling statistics if available
    rolling_features = ['rolling_3m_price_mean', 'price_yoy_change']
    rolling_features = [f for f in rolling_features if f in df_processed.columns]
    
    if rolling_features and 'sale_date' in df_processed.columns:
        for feature in rolling_features:
            # Calculate the average of the rolling statistic by month
            df_processed['year_month'] = df_processed['sale_date'].dt.to_period('M')
            monthly_avg = df_processed.groupby(['year_month', 'is_foreclosure'])[feature].mean().unstack().fillna(0)
            
            plt.figure(figsize=(14, 7))
            
            # Plot for both foreclosure and regular sales if available
            if 1 in monthly_avg.columns:
                plt.plot(monthly_avg.index.astype(str), monthly_avg[1], 
                         label='Foreclosure', marker='o', markersize=4, alpha=0.7)
            if 0 in monthly_avg.columns:
                plt.plot(monthly_avg.index.astype(str), monthly_avg[0], 
                         label='Regular', marker='x', markersize=4, alpha=0.7)
                
            plt.title(f'Monthly Average {feature} Trend')
            plt.xlabel('Month')
            plt.ylabel(feature)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.xticks(rotation=90)
            plt.tight_layout()
            plt.savefig(f'{dist_dir}/{feature}_monthly_trend.png')
            plt.close()
    
    # 7. Correlation heatmap for numeric features
    numeric_cols = ['living_area', 'rooms', 'floor_count', 'sale_year', 'sale_month', 'is_foreclosure']
    
    # Add new numeric features to correlation analysis
    if 'yearly_foreclosure_rate' in df_processed.columns:
        numeric_cols.append('yearly_foreclosure_rate')
    if 'rolling_3m_price_mean' in df_processed.columns:
        numeric_cols.append('rolling_3m_price_mean')
    if 'price_yoy_change' in df_processed.columns:
        numeric_cols.append('price_yoy_change')
    
    numeric_cols = [col for col in numeric_cols if col in df_processed.columns]
    
    if len(numeric_cols) > 1:  # Need at least 2 columns for correlation
        plt.figure(figsize=(12, 10))
        corr_matrix = df_processed[numeric_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
        plt.title('Correlation Matrix')
        plt.tight_layout()
        plt.savefig(f'{dist_dir}/correlation_heatmap.png')
        plt.close()

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
    elif contains_metrics:
        # If models_dict contains metrics, use them directly
        for name, metrics in models_dict.items():
            if isinstance(metrics, dict):
                # For RandomForest, we have a metrics dictionary
                results.append({
                    'Model': name,
                    'Accuracy': metrics.get('accuracy', 0),
                    'Precision': metrics.get('precision', 0),
                    'Recall': metrics.get('recall', 0),
                    'F1': metrics.get('f1_score', 0),
                    'ROC AUC': metrics.get('roc_auc', 0)
                })
            else:
                # For dummy classifiers, we might have a different format
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
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC']
        for metric in metrics:
            if metric in results_df.columns:
                plt.figure(figsize=(12, 6))
                sns.barplot(x='Model', y=metric, data=results_df)
                plt.title(f'{metric} Comparison Across Models')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(f'{output_dir}/{metric.lower().replace(" ", "_")}_comparison.png')
                plt.close()
    
    logging.info("Model comparison complete.")
    return results_df 