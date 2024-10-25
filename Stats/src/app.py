from flask import Flask, render_template, request, flash
import pandas as pd
import joblib
import numpy as np
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Load pre-trained model and scaler
model = joblib.load('models/logistic_regression_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Define the exact feature names used during training
FEATURE_NAMES = [
    'home_ppg', 'away_ppg', 
    'home_team_corner_count', 'away_team_corner_count',
    'home_team_shots', 'away_team_shots', 
    'home_team_shots_on_target', 'away_team_shots_on_target', 
    'home_team_possession', 'away_team_possession',
    'team_a_xg', 'team_b_xg'
]

@app.route('/')
def index():
    # Load CSV data to get available date range
    data = pd.read_csv('data/matches_agg.csv')
    data['date_GMT'] = pd.to_datetime(data['date_GMT'], errors='coerce')
    min_date = data['date_GMT'].min().strftime('%Y-%m-%d')
    max_date = data['date_GMT'].max().strftime('%Y-%m-%d')
    return render_template('index.html', min_date=min_date, max_date=max_date)

@app.route('/results', methods=['POST'])
def results():
    try:
        # Load CSV data
        data = pd.read_csv('data/matches_agg.csv')
        
        # Select all potentially needed columns including those for features and display
        columns_needed = FEATURE_NAMES + [
            'date_GMT',
            'home_team_name',
            'away_team_name',
            'home_team_goal_count',
            'away_team_goal_count'
        ]
        
        # Create DataFrame with only needed columns
        df = data[columns_needed].copy()
        
        # Convert date_GMT column to datetime and drop rows with NaT values
        df['date_GMT'] = pd.to_datetime(df['date_GMT'], errors='coerce')
        df = df.dropna(subset=['date_GMT'])

        # Get and validate date range
        start_date = pd.to_datetime(request.form.get('start_date'))
        end_date = pd.to_datetime(request.form.get('end_date'))
        end_date = end_date + pd.Timedelta(days=1)

        print(f"Date range in dataset: {df['date_GMT'].min()} to {df['date_GMT'].max()}")
        print(f"Requested date range: {start_date} to {end_date}")

        # Filter DataFrame based on the date range
        filtered_df = df[(df['date_GMT'] >= start_date) & (df['date_GMT'] <= end_date)]
        
        if filtered_df.empty:
            flash(f"No matches found between {start_date.strftime('%Y-%m-%d')} and {end_date.strftime('%Y-%m-%d')}.")
            return render_template('results.html', matches=[])

        print(f"Found {len(filtered_df)} matches in the selected date range")

        match_results = []
        for index, row in filtered_df.iterrows():
            try:
                # Extract features in the correct order
                features = []
                for feature_name in FEATURE_NAMES:
                    if pd.isna(row[feature_name]):
                        print(f"Warning: Missing value for feature {feature_name} in match {row['home_team_name']} vs {row['away_team_name']}")
                        features.append(0.0)  # or some other appropriate default value
                    else:
                        features.append(float(row[feature_name]))
                
                features = np.array([features])
                print(f"Features shape: {features.shape}")
                print(f"Feature names: {FEATURE_NAMES}")
                
                # Scale features
                features_scaled = scaler.transform(features)

                # Predict probabilities
                probabilities = model.predict_proba(features_scaled)

                match_results.append({
                    'date': row['date_GMT'].strftime('%Y-%m-%d'),
                    'home_team': row['home_team_name'],
                    'away_team': row['away_team_name'],
                    'home_goals': int(row['home_team_goal_count']),
                    'away_goals': int(row['away_team_goal_count']),
                    'probabilities': {
                        'home_win': f"{probabilities[0][2]*100:.1f}%",
                        'tie': f"{probabilities[0][1]*100:.1f}%",
                        'away_win': f"{probabilities[0][0]*100:.1f}%"
                    }
                })
            except Exception as e:
                print(f"Error processing match: {str(e)}")
                continue

        return render_template('results.html', matches=match_results)

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        flash(f"An error occurred while processing your request: {str(e)}")
        return render_template('results.html', matches=[])

if __name__ == '__main__':
    app.run(debug=True)