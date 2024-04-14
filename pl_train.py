import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score

class name_mapping(dict):
    __missing__ = lambda self, key: key

map_values = {
    "Brighton and Hove Albion": "Brighton",
    "Manchester United": "Manchester Utd",
    "Newcastle United": "Newcastle Utd",
    "Totenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "wolverhampton wanderers": "Wolves",
}
mapping = name_mapping(**map_values)

def rolling_avg(group, cols, new_cols):
    group = group.sort_values('date')
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

def predict(data, predictors):

    #split into train and test
    train = data[data['date'] < '2024-01-01']
    test = data[data['date'] > '2024-01-01']

    rf.fit(train[predictors], train['target'])
    predictions = rf.predict(test[predictors])

    #confusion matrix
    combined = pd.DataFrame(dict(actual=test['target'], predicted=predictions), index=test.index)
    error = precision_score(test['target'], predictions)

    #calculate accuracy
    accuracy = accuracy_score(test['target'], predictions)
    print('Accuracy: ', accuracy)
    return combined, error


dataset = pd.read_csv('pl_matches.csv', index_col=0)
dataset.columns = [c.lower() for c in dataset.columns]

#convert date into datetime
dataset['date'] = pd.to_datetime(dataset['date'])

#venue code home or away (0 for away and 1 for home)
dataset['venue_code'] = dataset['venue'].astype('category').cat.codes

#each opponent has their own code
dataset['opponent_code'] = dataset['opponent'].astype('category').cat.codes

#hour encoding
dataset['time_hour'] = dataset['time'].str.replace(':.+', '', regex=True).astype('int')

#day encoding
dataset['day_code'] = dataset['date'].dt.dayofweek

#create a target variable
dataset['target'] = (dataset['result'] == 'W').astype('int')

predictors = ['venue_code', 'opponent_code', 'day_code', 'time_hour']
cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt"]
new_cols = [f"{c}_rolling" for c in cols]

dataset_rolling = dataset.groupby("team").apply(lambda x: rolling_avg(x, cols, new_cols))
dataset_rolling = dataset_rolling.droplevel('team')
dataset_rolling.index = range(dataset_rolling.shape[0])

print(dataset_rolling[['opponent']])

rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)
combined, precision = predict(dataset_rolling, predictors + new_cols)

combined = combined.merge(dataset_rolling[['date', 'team', 'opponent','result']], left_index=True, right_index=True)

combined['new_team'] = combined['team'].map(mapping)

merged = combined.merge(combined, left_on=['date', 'new_team'], right_on=['date', 'opponent'])
print(merged[(merged['predicted_x'] == 1) & (merged['predicted_y'] == 0)]['actual_x'].value_counts()) 

