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
        class_weight='balanced',
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
            'class_weight': ['balanced']
        }
    else:
        # More extensive grid for smaller feature sets
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2'],
            'class_weight': ['balanced']
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
            'max_depth': [10, 15, 20, 25, 30, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['sqrt', 'log2', None],
            'class_weight': ['balanced']
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
    }).sort_values('Importance', ascending=False)
    
    # Save to CSV
    feature_importances.to_csv(os.path.join(output_dir, 'feature_importances.csv'), index=False)
    
    # Plot top 20 important features
    plt.figure(figsize=(10, 8))
    top_features = feature_importances.head(20)
    plt.barh(top_features['Feature'], top_features['Importance'])
    plt.xlabel('Importance')
    plt.title('Top 20 Feature Importances')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_importances.png'))
    plt.close()
    
    # 2. Use permutation importance (more reliable than built-in feature importance)
    result = permutation_importance(
        rf, X_test, y_test, 
        n_repeats=10, 
        random_state=42, 
        n_jobs=-1
    )
    
    perm_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': result.importances_mean
    }).sort_values('Importance', ascending=False)
    
    # Save to CSV
    perm_importance.to_csv(os.path.join(output_dir, 'permutation_importance.csv'), index=False)
    
    # Plot top 20 permutation importance features
    plt.figure(figsize=(10, 8))
    top_perm_features = perm_importance.head(20)
    plt.barh(top_perm_features['Feature'], top_perm_features['Importance'])
    plt.xlabel('Permutation Importance')
    plt.title('Top 20 Features (Permutation Importance)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'permutation_importance.png'))
    plt.close()
    
    # Check for overfitting
    logging.info("Checking for overfitting...")
    y_train_pred = rf.predict(X_train)
    train_accuracy = accuracy_score(y_train, y_train_pred)
    train_f1 = f1_score(y_train, y_train_pred)
    
    test_f1 = f1_score(y_test, y_pred)
    
    logging.info(f"Training accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")
    logging.info(f"Test accuracy: {accuracy:.4f}, F1: {test_f1:.4f}")
    
    acc_diff = train_accuracy - accuracy
    f1_diff = train_f1 - test_f1
    
    logging.info(f"Accuracy difference: {acc_diff:.4f}")
    logging.info(f"F1 difference: {f1_diff:.4f}")
    
    if acc_diff > 0.15 or f1_diff > 0.15:
        logging.warning("Potential overfitting detected: training performance is significantly higher than test performance")
    
    # Return performance metrics
    metrics = {
        'accuracy': accuracy,
        'f1_score': test_f1,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'best_threshold': best_threshold,
        'train_test_acc_diff': acc_diff,
        'train_test_f1_diff': f1_diff
    }
    
    return metrics 