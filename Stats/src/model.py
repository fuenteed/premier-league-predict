from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit

# Load the data
data = pd.read_csv('data/matches_agg.csv')

# Ensure the date column is in datetime format
data['date_GMT'] = pd.to_datetime(data['date_GMT'])  

# Sort data by date to ensure temporal ordering
data = data.sort_values('date_GMT')

# Features
X = data[[
    'home_ppg', 'away_ppg', 
    'home_team_corner_count', 'away_team_corner_count',
    'home_team_shots', 'away_team_shots', 
    'home_team_shots_on_target', 'away_team_shots_on_target', 
    'home_team_possession', 'away_team_possession',
    'team_a_xg', 'team_b_xg'
]]

# Prepare the predictor
def prepare_predictor(row):
    if row['home_team_goal_count'] > row['away_team_goal_count']:
        return 2  # home team win 
    elif row['home_team_goal_count'] < row['away_team_goal_count']:
        return 0  # away team win
    else:
        return 1  # tie

# Add the result feature to the dataset
data['result'] = data.apply(prepare_predictor, axis=1)
y = data['result']

# Time-based split
split_date = '2023-01-01'
train_mask = data['date_GMT'] < split_date
test_mask = data['date_GMT'] >= split_date

# Split the data
X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

# Print split information
print("\nData split information:")
print(f"Training data: {X_train.shape[0]} samples (matches before {split_date})")
print(f"Testing data: {X_test.shape[0]} samples (matches from {split_date} onwards)")

# Print class distribution for both sets
print("\nClass distribution in training set:")
print(y_train.value_counts(normalize=True))
print("\nClass distribution in test set:")
print(y_test.value_counts(normalize=True))

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Hyperparameter grids
param_grids = {
    'logistic': {
        'penalty': ['l2', 'l1', 'elasticnet'],
        'C': np.logspace(-4, 4, 20),
        'solver': ['lbfgs', 'newton-cg', 'sag', 'saga'],
        'max_iter': [100, 200, 500, 750, 1000, 1500]
    },
    'random_forest': {
        'n_estimators': [100, 200, 500],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['auto', 'sqrt', 'log2', None]
    },
    'xgboost': {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 2, 3],
        'gamma': [0, 0.1, 0.2],
    }
}


tscv = TimeSeriesSplit(n_splits=3)

# Logistic Regression
logistic_model = LogisticRegression()
logistic_grid = GridSearchCV(logistic_model, param_grids['logistic'], cv=tscv, verbose=True, n_jobs=-1)
logistic_grid.fit(X_train, y_train)
best_logistic = logistic_grid.best_estimator_
joblib.dump(best_logistic, 'models/logistic_model.pkl')

# Random Forest
rf_model = RandomForestClassifier(random_state=42)
rf_grid = GridSearchCV(rf_model, param_grids['random_forest'], cv=tscv, verbose=2, n_jobs=4)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_
joblib.dump(best_rf, 'models/random_forest_model.pkl')

# XGBoost
xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
xgb_grid = GridSearchCV(xgb_model, param_grids['xgboost'], cv=tscv, verbose=True, n_jobs=-1)
xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_
joblib.dump(best_xgb, 'models/xgboost_model.pkl')

# Print best parameters and accuracies
print("\nBest Parameters:")
print("Logistic Regression:", logistic_grid.best_params_)
print("Random Forest:", rf_grid.best_params_)
print("XGBoost:", xgb_grid.best_params_)

# Model evaluations
print("\nTest Set Accuracies:")
print("Logistic Regression:", accuracy_score(y_test, best_logistic.predict(X_test)))
print("Random Forest:", accuracy_score(y_test, best_rf.predict(X_test)))
print("XGBoost:", accuracy_score(y_test, best_xgb.predict(X_test)))

# Detailed evaluation of best model (assuming Random Forest)
best_model = best_rf  # Change this to best_xgb or best_logistic if they perform better
y_pred = best_model.predict(X_test)
probabilities = best_model.predict_proba(X_test)

print("\nDetailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Away Win', 'Draw', 'Home Win']))

# Create results DataFrame with dates
results_df = pd.DataFrame({
    'Date': data[test_mask]['date_GMT'],
    'Actual': y_test,
    'Predicted': y_pred,
    'Prob_Away_Win': probabilities[:, 0],
    'Prob_Draw': probabilities[:, 1],
    'Prob_Home_Win': probabilities[:, 2]
})

# Feature importance for Random Forest
if hasattr(best_model, 'feature_importances_'):
    print("\nFeature Importances:")
    for feature, importance in zip(X.columns, best_model.feature_importances_):
        print(f"{feature}: {importance:.3f}")

# Save the results and models
results_df.to_csv('data/predictions.csv', index=False)
joblib.dump(scaler, 'models/scaler.pkl')