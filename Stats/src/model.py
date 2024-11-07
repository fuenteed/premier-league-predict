import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import os

#Step 1: Load Data
data = pd.read_csv('data/matches_agg.csv')

#Step 2: Preprocess Data

# Convert date_GMT column to datetime and drop rows with NaT values
data['date_GMT'] = pd.to_datetime(data['date_GMT'], errors='coerce')
data = data.dropna(subset=['date_GMT'])

features_to_include = [
            'date_GMT',
            'home_team_name', 'away_team_name', 
            'home_team_goal_count','away_team_goal_count',
            'home_team_possession','away_team_possession',
            'team_a_xg', 'team_b_xg',
            'home_team_corner_count', 'away_team_corner_count',
             'home_team_shots', 'away_team_shots',
             'home_team_shots_on_target', 'away_team_shots_on_target',
]

# Create DataFrame with only needed columns
df = data[features_to_include].copy()

#grab the day, month, and year from the date
df['day'] = df['date_GMT'].dt.day
df['month'] = df['date_GMT'].dt.month
df['year'] = df['date_GMT'].dt.year


features_to_scale_and_train = [
            'home_team_possession','away_team_possession',
            'team_a_xg', 'team_b_xg',
            'home_team_corner_count', 'away_team_corner_count',
             'home_team_shots', 'away_team_shots',
             'home_team_shots_on_target', 'away_team_shots_on_target',
]

def prepare_target_variable(row):
    if row['home_team_goal_count'] > row['away_team_goal_count']:
        return 0
    elif row['home_team_goal_count'] < row['away_team_goal_count']:
        return 1
    else:
        return 2
    
df['target'] = df.apply(prepare_target_variable, axis=1)

#add a unique numberical code for each team 
temp = df[['home_team_name', 'away_team_name']].stack()
temp[:] = temp.factorize()[0]
df[['home_team_code', 'away_team_code']] = temp.unstack()

df.to_csv('data/matches_prep.csv', index=False)


#Step 3: Train Model

#Now load new preprocessed data
main_df = pd.read_csv('data/matches_prep.csv')

X = main_df.drop(columns=['target', 'home_team_name', 'away_team_name', 'home_team_goal_count', 'away_team_goal_count', 'date_GMT'])
y = df['target']

#Time Based SPlit
split_date = '2023-08-16'
train_mask = main_df['date_GMT'] < split_date
test_mask = main_df['date_GMT'] >= split_date

# Split the data
X_train = X[train_mask]
X_test = X[test_mask]
y_train = y[train_mask]
y_test = y[test_mask]

#Print split information
print("\nData split information:")
print(f"Training data: {X_train.shape[0]} samples (matches before {split_date})")
print(f"Testing data: {X_test.shape[0]} samples (matches from {split_date} onwards)")

# Print class distribution for both sets
"""
print("\nClass distribution in training set:")
print(y_train.value_counts(normalize=True))
print("\nClass distribution in test set:")
print(y_test.value_counts(normalize=True))
"""

print('\nTraining the forest model...')
#train the forest model
rf = RandomForestClassifier(random_state=1, n_estimators=100, min_samples_split=10, class_weight='balanced')
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)


print('\nTraining the xgboost model...')
#train the xgboost model
xgb = XGBClassifier(random_state=1, n_estimators=100, learning_rate=0.01, max_depth=6, eval_metric='mlogloss')
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)


#calculate accuracies and classification reports

print('\nAccuracy of Random Forest Model: ', accuracy_score(y_test, y_pred_rf))
combined_rf = pd.DataFrame(dict(actual=y_test, predicted=y_pred_rf))
print(pd.crosstab(index=combined_rf.actual, columns=combined_rf.predicted))


print('\nAccuracy of XGBoost Model: ', accuracy_score(y_test, y_pred_xgb))
combined_xgb = pd.DataFrame(dict(actual=y_test, predicted=y_pred_xgb))
print(pd.crosstab(index=combined_xgb.actual, columns=combined_xgb.predicted))


print('\nClassification Report of Random Forest Model: \n', classification_report(y_test, y_pred_rf))

print('\nClassification Report of XGBoost Model: \n', classification_report(y_test, y_pred_xgb))
