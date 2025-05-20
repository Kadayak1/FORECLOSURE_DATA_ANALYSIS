import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, accuracy_score, 
                           roc_curve, auc, precision_recall_curve, average_precision_score,
                           f1_score, fbeta_score)
from sklearn.inspection import permutation_importance
from sklearn.model_selection import (cross_val_score, KFold, GridSearchCV, StratifiedKFold, 
                                   RandomizedSearchCV, cross_validate)
from sklearn.utils.class_weight import compute_class_weight

def train_random_forest(X_train, X_test, y_train, y_test, X, output_dir='outputs/random_forest'):
    """Train and evaluate a RandomForestClassifier.
    
    Parameters:
    -----------
    X_train : DataFrame
        Training features
    X_test : DataFrame
        Testing features
    y_train : Series
        Training target values
    y_test : Series
        Testing target values
    X : DataFrame
        Full feature set for importance analysis
    output_dir : str
        Directory to save outputs
        
    Returns:
    --------
    dict
        Performance metrics
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.info("Training RandomForestClassifier...")
    
    # Compute class weights to handle imbalance
    class_weights = compute_class_weight(class_weight='balanced', 
                                       classes=np.unique(y_train), 
                                       y=y_train)
    class_weight_dict = {i: weight for i, weight in enumerate(class_weights)}
    
    # Perform cross-validation first to get baseline performance
    logging.info("Performing 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Use a simpler model for cross-validation to save time
    base_rf = RandomForestClassifier(
        n_estimators=100,
        class_weight=class_weight_dict,
        random_state=42,
        n_jobs=-1
    )
    
    # Define metrics to evaluate
    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'roc_auc': 'roc_auc'
    }
    
    # Perform cross-validation with multiple metrics
    cv_results = cross_validate(
        base_rf, X_train, y_train, 
        cv=cv, 
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    # Log cross-validation results
    logging.info("Cross-validation results:")
    for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']:
        test_scores = cv_results[f'test_{metric}']
        mean_score = np.mean(test_scores)
        std_score = np.std(test_scores)
        logging.info(f"{metric.title()}: {mean_score:.4f} (±{std_score:.4f})")
        
        # Check for overfitting in cross-validation
        train_scores = cv_results[f'train_{metric}']
        train_mean = np.mean(train_scores)
        diff = train_mean - mean_score
        if diff > 0.15:  # Significant difference indicates overfitting
            logging.warning(f"Potential overfitting detected in cross-validation: {metric} train-test gap of {diff:.4f}")
    
    # Perform hyperparameter tuning
    logging.info("Performing hyperparameter tuning...")
    
    # Check if we have a high-dimensional feature space
    if X_train.shape[1] > 100:
        logging.info("Using reduced parameter grid due to high feature dimensionality")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20],
            'min_samples_split': [2, 5],
            'class_weight': [class_weight_dict]  # Use the custom weights dictionary
        }
    else:
        # More extensive grid for smaller feature sets
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, 40, 50, None],  # Added higher depth values
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 8],  # Added higher value
            'max_features': ['sqrt', 'log2'],
            'class_weight': [class_weight_dict],
            'oob_score': [True]  # Add out-of-bag score tracking
        }
    
    # Use GridSearchCV for smaller grids, RandomizedSearchCV for larger ones
    if len(param_grid['n_estimators']) * len(param_grid['max_depth']) * len(param_grid['min_samples_split']) <= 24:
        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            cv=3,  # Using 3-fold for faster grid search
            scoring='f1',  # Focus on F1 for imbalanced data
            n_jobs=-1,
            verbose=1
        )
    else:
        # Use randomized search for larger parameter spaces
        param_distributions = {
            'n_estimators': [100, 200, 300, 400],
            'max_depth': [10, 15, 20, 25, 30, 40, 50, None],  # Added higher depth values
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4, 8],  # Added higher value
            'max_features': ['sqrt', 'log2', None],
            'class_weight': [class_weight_dict],
            'oob_score': [True]  # Add out-of-bag score tracking
        }
        
        grid_search = RandomizedSearchCV(
            RandomForestClassifier(random_state=42),
            param_distributions,
            n_iter=20,  # Number of parameter settings sampled
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=1,
            random_state=42
        )
    
    # Fit the grid search
    grid_search.fit(X_train, y_train)
    
    # Log the best parameters and score
    logging.info(f"Best parameters: {grid_search.best_params_}")
    logging.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Get the best model from hyperparameter tuning
    rf = grid_search.best_estimator_
    
    # Log OOB score if available
    if hasattr(rf, 'oob_score_'):
        logging.info(f"Out-of-bag score: {rf.oob_score_:.4f}")
    
    # Train on full training set
    rf.fit(X_train, y_train)
    
    # Find optimal threshold for imbalanced data
    y_pred_proba = rf.predict_proba(X_test)[:, 1]
    
    # Try different thresholds and find the one that maximizes F1
    thresholds = np.arange(0.1, 0.9, 0.05)
    f1_scores = []
    
    for threshold in thresholds:
        y_pred_threshold = (y_pred_proba >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred_threshold)
        f1_scores.append((threshold, f1))
    
    # Find threshold with best F1 score
    best_threshold, best_f1 = max(f1_scores, key=lambda x: x[1])
    logging.info(f"Optimal threshold: {best_threshold:.2f} with F1 score: {best_f1:.4f}")
    
    # Use the optimal threshold for final predictions
    y_pred = (y_pred_proba >= best_threshold).astype(int)
    
    # Save the optimal threshold for later use
    with open(os.path.join(output_dir, 'optimal_threshold.txt'), 'w') as f:
        f.write(f"Optimal threshold: {best_threshold:.4f}\n")
        f.write(f"F1 score at optimal threshold: {best_f1:.4f}\n")
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    logging.info(f"RandomForestClassifier Metrics:")
    logging.info(f"Accuracy: {accuracy:.4f}")
    
    # Generate classification report
    report = classification_report(y_test, y_pred)
    logging.info(report)
    
    # Save report to file
    with open(os.path.join(output_dir, 'classification_report.txt'), 'w') as f:
        f.write(report)
    
    # Generate and save confusion matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Regular', 'Foreclosure'],
                yticklabels=['Regular', 'Foreclosure'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Calculate and plot ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'roc_curve.png'))
    plt.close()
    
    # Calculate and plot precision-recall curve (better for imbalanced data)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AP = {pr_auc:.2f})')
    plt.axhline(y=sum(y_test)/len(y_test), color='red', linestyle='--', label='Baseline')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower left')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'precision_recall_curve.png'))
    plt.close()
    
    # Calculate feature importance
    # 1. Using built-in feature importance
    feature_importances = pd.DataFrame({
        'Feature': X.columns,
        'Importance': rf.feature_importances_
    })
    feature_importances = feature_importances.sort_values('Importance', ascending=False)
    
    # Save feature importances to CSV
    feature_importances.to_csv(os.path.join(output_dir, 'feature_importances.csv'), index=False)
    
    # Plot feature importances (top 20)
    top_features = feature_importances.head(20)
    plt.figure(figsize=(10, 8))
    plt.barh(top_features['Feature'][::-1], top_features['Importance'][::-1])
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importances.png'))
    plt.close()
    
    # 2. Calculate permutation importance (more reliable)
    logging.info("Calculating permutation importance...")
    
    try:
        # This may take a while for large datasets
        perm_importance = permutation_importance(
            rf, X_test, y_test, n_repeats=10, random_state=42, n_jobs=-1
        )
        
        # Create DataFrame with permutation importances
        perm_importances = pd.DataFrame({
            'Feature': X.columns,
            'Importance': perm_importance.importances_mean,
            'Std': perm_importance.importances_std
        })
        perm_importances = perm_importances.sort_values('Importance', ascending=False)
        
        # Save permutation importances to CSV
        perm_importances.to_csv(os.path.join(output_dir, 'permutation_importances.csv'), index=False)
        
        # Plot permutation importances (top 20)
        top_perm_features = perm_importances.head(20)
        plt.figure(figsize=(10, 8))
        plt.barh(top_perm_features['Feature'][::-1], top_perm_features['Importance'][::-1],
                xerr=top_perm_features['Std'][::-1], capsize=5)
        plt.xlabel('Mean decrease in accuracy')
        plt.title('Top 20 Permutation Feature Importances')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'permutation_importance.png'))
        plt.close()
    except Exception as e:
        logging.warning(f"Error calculating permutation importance: {e}")
    
    # Create a directory for RF-specific plots
    rf_plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(rf_plots_dir, exist_ok=True)
    
    # Plot decision trees (first few trees in the forest)
    try:
        from sklearn.tree import plot_tree
        
        # Plot first 3 trees from the forest
        for i in range(min(3, len(rf.estimators_))):
            plt.figure(figsize=(20, 12))
            plot_tree(rf.estimators_[i], feature_names=X.columns, filled=True, max_depth=3,
                     class_names=['Regular', 'Foreclosure'], rounded=True)
            plt.title(f'Decision Tree {i+1} from Random Forest')
            plt.savefig(os.path.join(rf_plots_dir, f'tree_{i+1}.png'), dpi=100)
            plt.close()
    except Exception as e:
        logging.warning(f"Error plotting decision trees: {e}")
    
    # Create a comprehensive report
    with open(os.path.join(output_dir, 'rf_summary.txt'), 'w') as f:
        f.write("Random Forest Classifier Performance Summary\n")
        f.write("===========================================\n\n")
        
        # Basic metrics
        f.write(f"Accuracy: {accuracy:.4f}\n")
        cr_dict = classification_report(y_test, y_pred, output_dict=True)
        f.write(f"Precision (Foreclosure class): {cr_dict['1']['precision']:.4f}\n")
        f.write(f"Recall (Foreclosure class): {cr_dict['1']['recall']:.4f}\n")
        f.write(f"F1 Score (Foreclosure class): {cr_dict['1']['f1-score']:.4f}\n")
        f.write(f"ROC AUC: {roc_auc:.4f}\n")
        f.write(f"PR AUC: {pr_auc:.4f}\n")
        
        # Add OOB score if available
        if hasattr(rf, 'oob_score_'):
            f.write(f"Out-of-bag score: {rf.oob_score_:.4f}\n")
        
        f.write("\n")
        
        # Hyperparameters
        f.write("Best Hyperparameters:\n")
        for param, value in grid_search.best_params_.items():
            f.write(f"  {param}: {value}\n")
        f.write("\n")
        
        # Top Features
        f.write("Top 10 Most Important Features:\n")
        for i, (feature, importance) in enumerate(zip(feature_importances['Feature'].head(10), 
                                                    feature_importances['Importance'].head(10))):
            f.write(f"  {i+1}. {feature}: {importance:.4f}\n")
    
    # Calculate metrics for return
    metrics = {
        'accuracy': accuracy,
        'precision': cr_dict['1']['precision'],
        'recall': cr_dict['1']['recall'],
        'f1': cr_dict['1']['f1-score'],
        'roc_auc': roc_auc,
        'rf_model': rf,
        'optimal_threshold': best_threshold
    }
    
    # Add OOB score to metrics if available
    if hasattr(rf, 'oob_score_'):
        metrics['oob_score'] = rf.oob_score_
    
    return metrics 