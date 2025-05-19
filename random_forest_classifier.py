import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV, StratifiedKFold

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
        Full feature set (for feature names)
    output_dir : str
        Directory to save outputs
        
    Returns:
    --------
    RandomForestClassifier
        Trained random forest model
    """
    logging.info("Training RandomForestClassifier...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # First run cross-validation to get a more robust performance estimate
    logging.info("Performing 5-fold cross-validation...")
    rf_cv = RandomForestClassifier(n_estimators=100, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Calculate multiple cross-validation metrics
    cv_accuracy = cross_val_score(rf_cv, X_train, y_train, cv=cv, scoring='accuracy')
    cv_precision = cross_val_score(rf_cv, X_train, y_train, cv=cv, scoring='precision')
    cv_recall = cross_val_score(rf_cv, X_train, y_train, cv=cv, scoring='recall')
    cv_f1 = cross_val_score(rf_cv, X_train, y_train, cv=cv, scoring='f1')
    cv_roc_auc = cross_val_score(rf_cv, X_train, y_train, cv=cv, scoring='roc_auc')
    
    logging.info("Cross-validation results:")
    logging.info(f"Accuracy: {cv_accuracy.mean():.4f} (±{cv_accuracy.std():.4f})")
    logging.info(f"Precision: {cv_precision.mean():.4f} (±{cv_precision.std():.4f})")
    logging.info(f"Recall: {cv_recall.mean():.4f} (±{cv_recall.std():.4f})")
    logging.info(f"F1 Score: {cv_f1.mean():.4f} (±{cv_f1.std():.4f})")
    logging.info(f"ROC AUC: {cv_roc_auc.mean():.4f} (±{cv_roc_auc.std():.4f})")
    
    # Save cross-validation results
    cv_results = pd.DataFrame({
        'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC AUC'],
        'Mean': [cv_accuracy.mean(), cv_precision.mean(), cv_recall.mean(), cv_f1.mean(), cv_roc_auc.mean()],
        'Std': [cv_accuracy.std(), cv_precision.std(), cv_recall.std(), cv_f1.std(), cv_roc_auc.std()]
    })
    cv_results.to_csv(f'{output_dir}/cross_validation_results.csv', index=False)
    
    # Now perform hyperparameter tuning
    logging.info("Performing hyperparameter tuning...")
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': [None, 'balanced']
    }
    
    # Use a smaller grid if there are many features
    if X_train.shape[1] > 20:
        logging.info("Using reduced parameter grid due to high feature dimensionality")
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [None, 20],
            'min_samples_split': [2, 5],
            'class_weight': [None, 'balanced']
        }
    
    # Use GridSearchCV with stratified k-fold
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=42),
        param_grid=param_grid,
        cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        scoring='f1',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    # Log best parameters
    logging.info(f"Best parameters: {grid_search.best_params_}")
    logging.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
    
    # Save best parameters
    pd.DataFrame([grid_search.best_params_]).to_csv(f'{output_dir}/best_parameters.csv', index=False)
    
    # Train final model with best parameters
    rf = RandomForestClassifier(random_state=42, **grid_search.best_params_)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    
    # Output RF metrics
    logging.info("RandomForestClassifier Metrics:")
    logging.info(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}")
    rf_report = classification_report(y_test, rf_pred)
    logging.info(rf_report)
    
    # Save report to file
    with open(f'{output_dir}/classification_report.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy_score(y_test, rf_pred):.4f}\n\n")
        f.write(rf_report)
    
    # Plot confusion matrix for RF
    rf_cm = confusion_matrix(y_test, rf_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues')
    plt.title('RandomForest Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    plt.close()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, rf_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Random Forest (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Random prediction curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Random Forest')
    plt.legend(loc="lower right")
    plt.savefig(f'{output_dir}/roc_curve.png')
    plt.close()
    
    # Precision-Recall curve (better for imbalanced datasets)
    precision, recall, _ = precision_recall_curve(y_test, rf_proba)
    avg_precision = average_precision_score(y_test, rf_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'Random Forest (AP = {avg_precision:.2f})')
    # Plot baseline
    baseline = sum(y_test) / len(y_test)
    plt.plot([0, 1], [baseline, baseline], 'k--', label=f'Baseline ({baseline:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend(loc="best")
    plt.savefig(f'{output_dir}/precision_recall_curve.png')
    plt.close()
    
    # Check for likely overfitting by comparing training and test performance
    logging.info("Checking for overfitting...")
    train_pred = rf.predict(X_train)
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, rf_pred)
    
    train_f1 = classification_report(y_train, train_pred, output_dict=True)['1']['f1-score']
    test_f1 = classification_report(y_test, rf_pred, output_dict=True)['1']['f1-score']
    
    logging.info(f"Training accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")
    logging.info(f"Test accuracy: {test_accuracy:.4f}, F1: {test_f1:.4f}")
    logging.info(f"Accuracy difference: {train_accuracy - test_accuracy:.4f}")
    logging.info(f"F1 difference: {train_f1 - test_f1:.4f}")
    
    if train_accuracy - test_accuracy > 0.05 or train_f1 - test_f1 > 0.05:
        logging.warning("Potential overfitting detected: training performance is significantly higher than test performance")
    elif test_accuracy > 0.9 and train_accuracy > 0.9:
        logging.warning("Unusually high accuracy on both training and test sets - possible data leakage or very easy problem")
        
    # Analyze feature importance
    analyze_feature_importance(rf, X, output_dir)
    
    # Permutation importance (more reliable than built-in feature importance)
    perm_importance = permutation_importance(rf, X_test, y_test, n_repeats=10, random_state=42)
    perm_imp_mean = perm_importance.importances_mean
    
    # Create DataFrame with permutation importances
    perm_imp_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': perm_imp_mean
    })
    perm_imp_df = perm_imp_df.sort_values('Importance', ascending=False)
    
    # Plot top permutation importances
    top_n = min(20, len(perm_imp_df))
    plt.figure(figsize=(12, 8))
    plt.barh(range(top_n), perm_imp_df['Importance'][:top_n], align='center')
    plt.yticks(range(top_n), perm_imp_df['Feature'][:top_n])
    plt.title('Permutation Feature Importance')
    plt.xlabel('Mean Decrease in Accuracy')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/permutation_importance.png')
    plt.close()
    
    # Save permutation importances to CSV
    perm_imp_df.to_csv(f'{output_dir}/permutation_importance.csv', index=False)
    
    return rf

def analyze_feature_importance(model, X, output_dir='outputs/random_forest'):
    """Analyze and plot feature importance from the model.
    
    Parameters:
    -----------
    model : RandomForestClassifier
        Trained model
    X : DataFrame
        Feature set with column names
    output_dir : str
        Directory to save outputs
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]  # Top 20 features
    
    top_n = min(20, len(indices))
    plt.figure(figsize=(12, 10))
    plt.title('Feature Importances')
    plt.barh(range(top_n), importances[indices[-top_n:]], align='center')
    plt.yticks(range(top_n), [X.columns[i] for i in indices[-top_n:]])
    plt.xlabel('Relative Importance')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/feature_importance.png')
    plt.close()
    
    # Save all importances to CSV
    importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': importances
    }).sort_values('Importance', ascending=False)
    
    importance_df.to_csv(f'{output_dir}/feature_importance.csv', index=False) 