import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import cross_val_score
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc

def train_dummy_classifiers(X_train, X_test, y_train, y_test, output_dir='outputs/dummy'):
    """Train and evaluate multiple DummyClassifier strategies.
    
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
    output_dir : str
        Directory to save outputs
        
    Returns:
    --------
    dict
        Dictionary of trained dummy classifiers
    """
    logging.info("Training baseline models (DummyClassifiers)...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    strategies = ['most_frequent', 'stratified', 'uniform', 'prior']
    results = {}
    
    # Create a DataFrame to store metrics
    metrics_df = pd.DataFrame(columns=['Strategy', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC AUC'])
    
    # Plot setup for ROC curves
    plt.figure(figsize=(10, 8))
    
    for strategy in strategies:
        logging.info(f"Training DummyClassifier with strategy: {strategy}")
        dummy = DummyClassifier(strategy=strategy, random_state=42)
        
        # Cross-validation
        cv_scores = cross_val_score(dummy, X_train, y_train, cv=5)
        logging.info(f"Cross-validation scores ({strategy}): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Train on the full training set
        dummy.fit(X_train, y_train)
        
        # Predict
        y_pred = dummy.predict(X_test)
        
        # For ROC curve and AUC
        try:
            y_proba = dummy.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{strategy} (AUC = {roc_auc:.2f})')
        except:
            roc_auc = 0.5  # Default for strategies that don't support predict_proba
            logging.info(f"Strategy {strategy} doesn't support predict_proba, using default AUC of 0.5")
        
        # Classification report
        cr = classification_report(y_test, y_pred, output_dict=True)
        
        # Store metrics
        metrics_df = pd.concat([metrics_df, pd.DataFrame({
            'Strategy': [strategy],
            'Accuracy': [accuracy_score(y_test, y_pred)],
            'Precision': [cr['1']['precision']],
            'Recall': [cr['1']['recall']],
            'F1': [cr['1']['f1-score']],
            'ROC AUC': [roc_auc]
        })], ignore_index=True)
        
        # Save confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {strategy}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(f'{output_dir}/confusion_matrix_{strategy}.png')
        plt.close()
        
        results[strategy] = dummy
    
    # Finalize ROC curve plot
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Different Dummy Strategies')
    plt.legend(loc="lower right")
    plt.savefig(f'{output_dir}/roc_curves.png')
    plt.close()
    
    # Save metrics to CSV
    metrics_df.to_csv(f'{output_dir}/metrics_comparison.csv', index=False)
    
    # Also save as a pretty table image
    plt.figure(figsize=(10, 4))
    plt.axis('off')
    
    # Convert the DataFrame values to a numpy array and format it correctly
    table_data = metrics_df.copy()
    # Convert numeric columns to strings with 3 decimal places
    for col in table_data.columns:
        if col != 'Strategy':  # Skip the Strategy column which is a string
            table_data[col] = table_data[col].round(3).astype(str)
    
    table = plt.table(
        cellText=table_data.values,
        colLabels=metrics_df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    plt.savefig(f'{output_dir}/metrics_table.png', bbox_inches='tight', dpi=300)
    plt.close()
    
    logging.info("Dummy classifiers evaluation complete.")
    return results

def train_logistic_regression(X_train, X_test, y_train, y_test, X, output_dir='outputs/logistic'):
    """Train and evaluate a Logistic Regression classifier as an interpretable baseline.
    
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
    LogisticRegression
        Trained logistic regression model
    """
    from sklearn.linear_model import LogisticRegression
    
    logging.info("Training Logistic Regression model...")
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize and train model
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    
    # Predictions
    y_pred = lr.predict(X_test)
    y_proba = lr.predict_proba(X_test)[:, 1]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    
    logging.info(f"Logistic Regression Accuracy: {accuracy:.4f}")
    logging.info(f"Logistic Regression Classification Report:\n{cr}")
    
    # Save report to file
    with open(f'{output_dir}/classification_report.txt', 'w') as f:
        f.write(f"Accuracy: {accuracy:.4f}\n\n")
        f.write(cr)
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'Logistic Regression (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')  # Random prediction curve
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Logistic Regression')
    plt.legend(loc="lower right")
    plt.savefig(f'{output_dir}/roc_curve.png')
    plt.close()
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Logistic Regression Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'{output_dir}/confusion_matrix.png')
    plt.close()
    
    # Model coefficients
    feature_names = X.columns
    coefficients = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': lr.coef_[0]
    })
    coefficients = coefficients.sort_values('Coefficient', ascending=False)
    
    # Save top/bottom coefficients
    top_n = 20
    top_coef = coefficients.head(top_n)
    bottom_coef = coefficients.tail(top_n)
    
    # Plot coefficients
    plt.figure(figsize=(12, 10))
    
    plt.subplot(2, 1, 1)
    plt.barh(range(top_n), top_coef['Coefficient'], align='center')
    plt.yticks(range(top_n), top_coef['Feature'])
    plt.title('Top Positive Coefficients')
    plt.tight_layout()
    
    plt.subplot(2, 1, 2)
    plt.barh(range(top_n), bottom_coef['Coefficient'], align='center')
    plt.yticks(range(top_n), bottom_coef['Feature'])
    plt.title('Top Negative Coefficients')
    plt.tight_layout()
    
    plt.savefig(f'{output_dir}/coefficients.png')
    plt.close()
    
    # Save all coefficients to CSV
    coefficients.to_csv(f'{output_dir}/coefficients.csv', index=False)
    
    return lr 