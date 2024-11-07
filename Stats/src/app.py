from flask import Flask, render_template, request, flash, send_from_directory
import pandas as pd
import pickle
import os
from model import MatchModel
from datetime import datetime
import logging
import matplotlib
import numpy as np
matplotlib.use('Agg')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = 'supersecretkey'


models = {
    'forest': 'Forest',
    'bayes': 'Bayes'
}

# Define the exact feature names used during training
FEATURE_NAMES = [
    'home_team_code', 'away_team_code',
    'home_team_corner_count', 'away_team_corner_count',
    'home_team_shots', 'away_team_shots', 
    'home_team_shots_on_target', 'away_team_shots_on_target', 
    'home_team_possession', 'away_team_possession',
    'team_a_xg', 'team_b_xg'
]


@app.route('/')
def index():

    # Load data
    data = pd.read_csv('data/matches_agg.csv')

    """Render the main page"""
    try:
        if data is None:
            flash("Error: Dataset not available")
            return render_template('index.html', min_date=None, max_date=None)
            
        min_date = data['date_GMT'].min()
        max_date = data['date_GMT'].max()
        
        return render_template('index.html', 
                             min_date=min_date.strftime('%Y-%m-%d'),
                             max_date=max_date.strftime('%Y-%m-%d'))
                             
    except Exception as e:
        logger.error(f"Error in index route: {e}")
        flash("An error occurred while loading the page")
        return render_template('index.html', min_date=None, max_date=None)


@app.route('/results', methods=['POST'])
def results():
    """Process prediction request and show results"""
    try:
        #get model choice from user
        model_name = request.form.get('model')

        # set model of choice to forest or bayes
        if(model_name == 'forest'):
            model_choice = 'forest'
        elif(model_name == 'bayes'):
            model_choice = 'bayes'

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
                


                #predict the result, result should be either 0 or 1 or 2
                result = model_choice.predict(features)

                
                # Predict probabilities
                probabilities = model_choice.predict_proba(features)

                match_results.append({
                    'date': row['date_GMT'].strftime('%Y-%m-%d'),
                    'home_team': row['home_team_name'],
                    'away_team': row['away_team_name'],
                    'home_goals': int(row['home_team_goal_count']),
                    'away_goals': int(row['away_team_goal_count']),
                    'result': result[0],
                    'probabilities': {
                        'home_win': f"{probabilities[0][2]*100:.1f}%",
                        'tie': f"{probabilities[0][1]*100:.1f}%",
                        'away_win': f"{probabilities[0][0]*100:.1f}%"
                    }
                })
            except Exception as e:
                print(f"Error processing match: {str(e)}")
                continue

        #return the results and whether the model predicted 0 or 1 or 2
        return render_template('results.html', matches=match_results, model = model_choice)

    except Exception as e:
        print(f"Error processing request: {str(e)}")
        flash(f"An error occurred while processing your request: {str(e)}")
        return render_template('results.html', matches=[])

if __name__ == '__main__':
    app.run(debug=True)