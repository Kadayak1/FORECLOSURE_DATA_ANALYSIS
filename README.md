# FORECLOSURE DATA ANALYSIS

## Project Overview
This project analyzes foreclosure data from the Region Hovedstaden in Denmark (Copenhagen, Frederiksberg, etc.) to identify patterns and predict properties at risk of foreclosure using machine learning models. By comparing regular property sales with foreclosure sales, we aim to uncover underlying factors that may contribute to foreclosure risk based on property characteristics.

## Time-Based Features
The models include several time-based features to improve prediction performance:

1. **Yearly Foreclosure Rates**: Annual foreclosure rates are calculated and used as a feature to capture historical foreclosure trends.

2. **Economic Era Binning**: Time periods are categorized into economic eras to capture different market conditions:
   - Pre-Financial Crisis (before 2008)
   - Financial Crisis (2008-2012)
   - Recovery Period (2013-2019)
   - COVID Period (2020-2021)
   - Post-COVID (2022 onwards)

3. **Rolling Statistics**: 
   - 3-month rolling average of property prices by postal region
   - Year-on-year percentage changes in prices where sufficient historical data exists

## Feature Sets
The analysis compares four different feature set configurations to understand their impact on model performance:

1. **Full**: Uses all available features including temporal and price-related features.
2. **No_temporal**: Excludes temporal features (sale_year, sale_month, sale_quarter, sale_day_of_week, economic_era) but keeps price features.
3. **No_Price**: Excludes price-related features (Price, rolling_3m_price_mean, price_yoy_change) but keeps temporal features.
4. **No_temporal_No_price**: Excludes both temporal and price-related features, focusing only on property characteristics.

Each feature set is evaluated separately to understand the predictive power of different types of features.

## ML Algorithm Explanation

### Implemented Models

1. **Dummy Classifier (Baseline)**
   - Provides a simple baseline performance by employing strategies such as:
     - `most_frequent`: Always predicts the most frequent class in training data
     - `stratified`: Generates predictions based on the training set's class distribution
     - `uniform`: Generates predictions uniformly at random
     - `prior`: Generates predictions based on the class prior probabilities
   - Used to establish minimum performance thresholds for more complex models

2. **Random Forest Classifier**
   - Ensemble learning method that constructs multiple decision trees during training
   - Handles class imbalance using balanced class weights
   - Implements hyperparameter tuning to find optimal model configuration
   - Features optimal threshold selection for imbalanced data classification
   - Provides feature importance analysis to identify key predictive variables

### Feature Engineering & Selection

The code implements several feature set configurations to evaluate different aspects of the data:

1. **Full**: All available features including property characteristics, temporal patterns, and price trends
2. **No Temporal**: Excludes time-based features to assess if property characteristics alone are predictive
3. **No Price**: Excludes price-related features to avoid potential data leakage
4. **No Price & No Temporal**: Uses only core property characteristics as predictors

Each configuration helps isolate the impact of different feature types on foreclosure prediction.

## Code Workflow

1. **Data Preprocessing**
   - Feature engineering to create derived features from raw property data
   - Missing value imputation with appropriate strategies based on feature type
   - Encoding of categorical variables
   - Creation of temporal and price-related features

2. **Model Training & Evaluation**
   - Cross-validation to ensure robust model assessment
   - Hyperparameter tuning for the Random Forest model
   - Calculation of multiple performance metrics: Accuracy, Precision, Recall, F1 Score, ROC AUC
   - Optimal classification threshold selection for imbalanced data

3. **Result Analysis & Visualization**
   - Feature importance analysis to identify key predictors
   - Comparative analysis across different feature sets
   - Visualization of model performance metrics
   - Evaluation of different strategies for the baseline model

## Dependencies

The project requires the following Python packages:
```
pandas>=1.3.0       # Data manipulation and analysis
numpy>=1.20.0       # Numerical computing
scikit-learn>=1.0.0 # Machine learning algorithms and metrics
matplotlib>=3.4.0   # Data visualization
seaborn>=0.11.0     # Enhanced visualization
```

## Running the Analysis
To run the complete analysis:
```python
python analyze_ml_ready_data.py
```

You will be prompted to select which models to run from the following options:
1. Run Enhanced Dummy Classifiers
2. Run Random Forest Classifier
3. Run both classifiers
4. Debug data preprocessing

## Expected Outputs

The analysis generates comprehensive outputs in several directories:

### Evaluation Metrics
- **Accuracy**: Overall percentage of correct predictions
- **Precision**: How many of the predicted foreclosures were actually foreclosures
- **Recall**: How many actual foreclosures were correctly identified
- **F1 Score**: Harmonic mean of precision and recall, balancing both concerns
- **ROC AUC**: Area under Receiver Operating Characteristic curve, measuring discrimination ability

### Output Directories
- `outputs/general/`: General data analysis, distributions, correlations
- `outputs/dummy/`: Performance metrics for baseline models
- `outputs/random_forest/`: Complete Random Forest analysis including:
  - Classification reports
  - Confusion matrices
  - ROC curves
  - Precision-recall curves
  - Feature importance rankings
  - Model parameters
- `outputs/comparison/`: Model comparison within each feature set
- `outputs/overall_comparison/`: Cross-feature set performance comparison

### Visualization Examples
- Bar charts comparing F1 Score, Accuracy, Precision, Recall, and ROC AUC across models
- ROC curves showing trade-offs between true positive and false positive rates
- Feature importance plots highlighting key predictors of foreclosure
- Confusion matrices showing the breakdown of model predictions

The primary goal is to identify whether property attributes alone can predict foreclosure risk, or whether temporal and price information significantly improves prediction capability.
