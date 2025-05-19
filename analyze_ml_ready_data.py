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
    df['sale_date'] = pd.to_datetime(df['Sale Date'], format='%d-%m-%Y', errors='coerce')
    
    # Examine foreclosure distribution over time
    foreclosure_by_year = df.groupby([df['is_foreclosure'], df['sale_date'].dt.year])['Property ID'].count().unstack(0)
    
    # Plot historical trend
    plt.figure(figsize=(12, 6))
    if 1 in foreclosure_by_year.columns:
        plt.plot(foreclosure_by_year.index, foreclosure_by_year[1], 
                label='Foreclosures', marker='o', linewidth=2)
    if 0 in foreclosure_by_year.columns:
        plt.plot(foreclosure_by_year.index, foreclosure_by_year[0], 
                label='Regular Sales', marker='x', linewidth=2)
    plt.title('Sales by Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Sales')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig('outputs/general/temporal_distribution.png')
    
    # Examine property type distributions
    prop_types = pd.crosstab(df['is_foreclosure'], df['Property_Type'])
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
        class_balance = df['is_foreclosure'].value_counts(normalize=True)
        f.write(f"Class balance:\n{class_balance.to_string()}\n\n")
        
        # Property type distribution
        f.write(f"Property types by source:\n{prop_types.to_string()}\n\n")
        f.write(f"Property types percentage by source:\n{prop_pct.round(1).to_string()}\n\n")
        
        # Temporal ranges
        f.write("Date ranges:\n")
        for value in df['is_foreclosure'].unique():
            min_date = df[df['is_foreclosure'] == value]['sale_date'].min()
            max_date = df[df['is_foreclosure'] == value]['sale_date'].max()
            f.write(f"{value}: {min_date} to {max_date}\n")
        
        # Missing values by source
        f.write("\nMissing values by source:\n")
        for value in df['is_foreclosure'].unique():
            src_df = df[df['is_foreclosure'] == value]
            f.write(f"\n{value}:\n{src_df.isnull().sum().to_string()}\n")
    
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
    
    # Create a dictionary to store all model results
    all_results = {}
    
    # Define the feature sets to evaluate
    feature_sets = ['full', 'no_temporal', 'no_price', 'no_price_temporal']
    
    # Create the model-based directories
    if choice in [1, 3]:
        os.makedirs('outputs/dummy', exist_ok=True)
        for feature_set in feature_sets:
            os.makedirs(f'outputs/dummy/{feature_set}', exist_ok=True)
    
    if choice in [2, 3]:
        os.makedirs('outputs/random_forest', exist_ok=True)
        for feature_set in feature_sets:
            os.makedirs(f'outputs/random_forest/{feature_set}', exist_ok=True)
    
    # Clean up old directory structure if it exists
    for feature_set in feature_sets:
        if os.path.exists(f'outputs/{feature_set}'):
            import shutil
            try:
                shutil.rmtree(f'outputs/{feature_set}')
                logging.info(f"Removed old directory: outputs/{feature_set}")
            except Exception as e:
                logging.warning(f"Could not remove old directory structure: {e}")

    # Run each feature set configuration
    for feature_set in feature_sets:
        logging.info(f"\n\n{'='*50}")
        logging.info(f"EVALUATING FEATURE SET: {feature_set}")
        logging.info(f"{'='*50}\n")
        
        # Prepare features with the current feature set configuration
        if feature_set == 'no_price':
            # For 'no_price', we want to exclude price features but keep temporal features
            X, y, df_processed = prepare_features(df, check_leakage=True, remove_price_features=True, feature_set='full')
        else:
            X, y, df_processed = prepare_features(df, check_leakage=True, remove_price_features=False, feature_set=feature_set)
        
        # Log class balance
        logging.info(f"Class balance: {y.mean()*100:.2f}% foreclosures, {(1-y.mean())*100:.2f}% regular sales")
        
        # Create a directory for distributions
        distributions_dir = f'outputs/distributions/{feature_set}'
        os.makedirs(distributions_dir, exist_ok=True)
        
        # Plot feature distributions for this feature set
        plot_data_distributions(df_processed, output_dir=distributions_dir)
        
        # Train-test split with stratification to maintain class balance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        models_performance = {}
        
        # Run model(s) based on user choice
        if choice in [1, 3]:
            # Train and evaluate dummy classifiers
            dummy_output_dir = f'outputs/dummy/{feature_set}'
            dummy_metrics = train_dummy_classifiers(X_train, X_test, y_train, y_test, output_dir=dummy_output_dir)
            models_performance['dummy'] = dummy_metrics
        
        if choice in [2, 3]:
            # Train and evaluate random forest
            rf_output_dir = f'outputs/random_forest/{feature_set}'
            rf_metrics = train_random_forest(X_train, X_test, y_train, y_test, X, output_dir=rf_output_dir)
            models_performance['random_forest'] = rf_metrics
        
        # If running both models, compare their performance
        if choice == 3:
            # Create comparison directory
            comparison_dir = f'outputs/comparison/{feature_set}'
            os.makedirs(comparison_dir, exist_ok=True)
            compare_models(models_performance, X_test, y_test, output_dir=comparison_dir)
        
        # Store results for this feature set
        all_results[feature_set] = models_performance
    
    # Create a directory for overall comparison
    overall_comparison_dir = 'outputs/overall_comparison'
    if not os.path.exists(overall_comparison_dir):
        os.makedirs(overall_comparison_dir)
    
    # Compare results across all feature sets
    if len(all_results) > 0:
        logging.info("\n\n" + "="*80)
        logging.info("FINAL PERFORMANCE COMPARISON ACROSS FEATURE SETS AND MODELS")
        logging.info("="*80 + "\n")
        
        # Create organized comparison tables
        
        # 1. First, organize by model type
        model_types = ['random_forest', 'dummy']
        if choice in [1, 3]:
            # For each model type, create a table comparing feature sets
            for model_type in model_types:
                if model_type not in all_results[list(all_results.keys())[0]]:
                    continue  # Skip if this model wasn't run
                
                if model_type == 'random_forest':
                    model_name = "Random Forest"
                else:
                    model_name = "Dummy Classifiers (Baseline)"
                
                logging.info(f"\n{'-'*20} {model_name} Performance {'-'*20}")
                
                # Create a table for this model type
                comparison_table = []
                for feature_set, models in all_results.items():
                    if model_type in models:
                        metrics = models[model_type]
                        if model_type == 'random_forest':
                            # For Random Forest, we have a metrics dictionary
                            comparison_table.append({
                                'Feature Set': feature_set,
                                'Accuracy': metrics['accuracy'],
                                'Precision': metrics['precision'],
                                'Recall': metrics['recall'],
                                'F1 Score': metrics['f1'],
                                'ROC AUC': metrics['roc_auc']
                            })
                        elif model_type == 'dummy':
                            # For dummy classifiers, we have dummy classifier objects
                            # Get the metrics from the most_frequent strategy (common baseline)
                            dummy_metrics = {}
                            for strategy, clf in metrics.items():
                                if strategy == 'most_frequent':
                                    try:
                                        y_pred = clf.predict(X_test)
                                        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                                        
                                        dummy_metrics['accuracy'] = accuracy_score(y_test, y_pred)
                                        dummy_metrics['precision'] = precision_score(y_test, y_pred, zero_division=0)
                                        dummy_metrics['recall'] = recall_score(y_test, y_pred, zero_division=0)
                                        dummy_metrics['f1'] = f1_score(y_test, y_pred, zero_division=0)
                                        
                                        # Try to get ROC AUC if possible
                                        try:
                                            y_proba = clf.predict_proba(X_test)[:, 1]
                                            dummy_metrics['roc_auc'] = roc_auc_score(y_test, y_proba)
                                        except:
                                            dummy_metrics['roc_auc'] = 0.5
                                    except:
                                        # Use generic values if prediction fails
                                        dummy_metrics = {
                                            'accuracy': 0.0,
                                            'precision': 0.0,
                                            'recall': 0.0,
                                            'f1': 0.0,
                                            'roc_auc': 0.5
                                        }
                            
                            comparison_table.append({
                                'Feature Set': feature_set,
                                'Accuracy': dummy_metrics.get('accuracy', 0.0),
                                'Precision': dummy_metrics.get('precision', 0.0),
                                'Recall': dummy_metrics.get('recall', 0.0),
                                'F1 Score': dummy_metrics.get('f1', 0.0),
                                'ROC AUC': dummy_metrics.get('roc_auc', 0.5)
                            })
                
                # Convert to DataFrame and print
                comparison_df = pd.DataFrame(comparison_table)
                
                # Sort by feature set to ensure consistent order
                feature_set_order = {fs: i for i, fs in enumerate(feature_sets)}
                comparison_df['sort_order'] = comparison_df['Feature Set'].map(feature_set_order)
                comparison_df = comparison_df.sort_values('sort_order').drop('sort_order', axis=1)
                
                # Print the table
                logging.info(comparison_df.to_string(index=False, float_format='{:.4f}'.format))
                
                # Save to CSV in overall comparison directory
                comparison_df.to_csv(f'{overall_comparison_dir}/{model_type}_comparison.csv', index=False)
        
        # 2. Then, organize by feature set
        logging.info(f"\n{'-'*20} Performance By Feature Set {'-'*20}")
        for feature_set in feature_sets:
            if feature_set not in all_results:
                continue
            
            logging.info(f"\nFeature Set: {feature_set}")
            
            # Create a table comparing models for this feature set
            comparison_table = []
            
            for model_type in model_types:
                if model_type not in all_results[feature_set]:
                    continue
                
                metrics = all_results[feature_set][model_type]
                
                if model_type == 'random_forest':
                    # For Random Forest, we have a metrics dictionary
                    comparison_table.append({
                        'Model': 'Random Forest',
                        'Accuracy': metrics['accuracy'],
                        'Precision': metrics['precision'],
                        'Recall': metrics['recall'],
                        'F1 Score': metrics['f1'],
                        'ROC AUC': metrics['roc_auc']
                    })
                elif model_type == 'dummy':
                    # For dummy classifiers, we have dummy classifier objects
                    # We'll add each strategy as a separate row
                    for strategy, clf in metrics.items():
                        try:
                            y_pred = clf.predict(X_test)
                            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                            
                            accuracy = accuracy_score(y_test, y_pred)
                            precision = precision_score(y_test, y_pred, zero_division=0)
                            recall = recall_score(y_test, y_pred, zero_division=0)
                            f1 = f1_score(y_test, y_pred, zero_division=0)
                            
                            # Try to get ROC AUC if possible
                            try:
                                y_proba = clf.predict_proba(X_test)[:, 1]
                                roc_auc = roc_auc_score(y_test, y_proba)
                            except:
                                roc_auc = 0.5
                            
                            comparison_table.append({
                                'Model': f'Dummy ({strategy})',
                                'Accuracy': accuracy,
                                'Precision': precision,
                                'Recall': recall,
                                'F1 Score': f1,
                                'ROC AUC': roc_auc
                            })
                        except:
                            # Skip if prediction fails
                            pass
            
            # Convert to DataFrame and print
            comparison_df = pd.DataFrame(comparison_table)
            if not comparison_df.empty:
                logging.info(comparison_df.to_string(index=False, float_format='{:.4f}'.format))
                
                # Save to CSV in comparison directory
                comparison_df.to_csv(f'outputs/comparison/{feature_set}/model_comparison.csv', index=False)
        
        # Create a side-by-side bar chart for Random Forest performance across feature sets
        rf_comparison = []
        for feature_set, models in all_results.items():
            if 'random_forest' in models:
                metrics = models['random_forest']
                rf_comparison.append({
                    'Feature Set': feature_set,
                    'Accuracy': metrics['accuracy'],
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1 Score': metrics['f1'],
                    'ROC AUC': metrics['roc_auc']
                })
        
        # Convert to DataFrame and save
        rf_comparison_df = pd.DataFrame(rf_comparison)
        rf_comparison_df.to_csv(f'{overall_comparison_dir}/random_forest_comparison.csv', index=False)
        
        # Create a bar chart for key metrics
        plt.figure(figsize=(12, 8))
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC']
        
        # Get the feature sets and values for each metric
        feature_sets = rf_comparison_df['Feature Set'].tolist()
        
        # Create more descriptive labels
        display_labels = []
        for fs in feature_sets:
            if fs == 'full':
                display_labels.append('Full')
            elif fs == 'no_temporal':
                display_labels.append('No Temporal')
            elif fs == 'no_price':
                display_labels.append('No Price')
            elif fs == 'no_price_temporal':
                display_labels.append('No Price & No Temporal')
            else:
                display_labels.append(fs)
        
        # Set width of bars
        bar_width = 0.15
        positions = np.arange(len(feature_sets))
        
        # Plotting each metric
        for i, metric in enumerate(metrics):
            plt.bar(positions + i*bar_width, rf_comparison_df[metric], 
                   width=bar_width, label=metric)
        
        # Add labels and legend
        plt.xlabel('Feature Set')
        plt.ylabel('Score')
        plt.title('Random Forest Performance Across Feature Sets')
        plt.xticks(positions + bar_width*2, display_labels, rotation=15)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'{overall_comparison_dir}/feature_set_comparison.png')
        
        logging.info("\nComparison chart saved to " + f'{overall_comparison_dir}/feature_set_comparison.png')
    
    logging.info("\nAnalysis complete. Check the output folders for results.")

if __name__ == "__main__":
    main() 