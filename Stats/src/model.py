from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib

# Load the data
data = pd.read_csv('data/matches_agg.csv')

# Features - removed goal counts from features
X = data[[
    'home_ppg', 'away_ppg', 
    'home_team_corner_count', 'away_team_corner_count',
    'home_team_shots', 'away_team_shots', 
    'home_team_shots_on_target', 'away_team_shots_on_target', 
    'home_team_possession', 'away_team_possession',
    'team_a_xg', 'team_b_xg'
]]

def prepare_predictor(row):
    if row['home_team_goal_count'] > row['away_team_goal_count']:
        return 2  # home team win (changed to match your app's probability indexing)
    elif row['home_team_goal_count'] < row['away_team_goal_count']:
        return 0  # away team win
    else:
        return 1  # tie

# Add the result feature to the dataset
data['result'] = data.apply(prepare_predictor, axis=1)
y = data['result']

# Print class distribution
print("\nClass distribution:")
print(y.value_counts(normalize=True))

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Normalize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the logistic regression model
model = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    max_iter=1000,
    class_weight='balanced'  # Add class weights to handle imbalanced data
)
model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = model.predict(X_test_scaled)
probabilities = model.predict_proba(X_test_scaled)

# Print detailed metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Away Win', 'Draw', 'Home Win']))

# Create a DataFrame with actual values, predictions, and probabilities
results_df = pd.DataFrame({
    'Actual': y_test.values,
    'Predicted': y_pred,
    'Prob_Away_Win': probabilities[:, 0],
    'Prob_Draw': probabilities[:, 1],
    'Prob_Home_Win': probabilities[:, 2]
})

# Save the first few predictions for inspection
print("\nSample predictions:")
print(results_df.head())

# Calculate feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': abs(model.coef_).mean(axis=0)
})
print("\nFeature Importance:")
print(feature_importance.sort_values('Importance', ascending=False))

# Save the results and models
results_df.to_csv('data/predictions.csv', index=False)
joblib.dump(model, 'models/logistic_regression_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')

# Print final accuracy
print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.3f}")