# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 18:49:24 2017

@author: Jeff
"""
import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# import the datasets
df_seeds = pd.read_csv('Datasets/TourneySeeds.csv')
df_tour = pd.read_csv('Datasets/TourneyCompactResults.csv')

# drop labels we aren't using from df_tour
df_tour.drop(labels=['Daynum', 'Wscore', 'Lscore', 'Wloc', 'Numot'], inplace=True, axis=1)

# function we will use to obtain the seed integer from df_seeds table
def seed_to_int(seed):
    """Get just the digits from the seeding. Return as int"""
    s_int = int(seed[1:3])
    return s_int

# apply the function and drop the original Seed column
df_seeds['n_seed'] = df_seeds.Seed.apply(seed_to_int)
df_seeds.drop(labels=['Seed'], inplace=True, axis=1)

# divide the data into winners and losers
df_winseeds = df_seeds.rename(columns={'Team':'Wteam', 'n_seed':'win_seed'})
df_lossseeds = df_seeds.rename(columns={'Team':'Lteam', 'n_seed':'loss_seed'})

# use a dummy df to merge winner / loser dfs with tourney results
# add a seed_diff column to indicate the discrepancy between seeds
df_dummy = pd.merge(left=df_tour, right=df_winseeds, how='left', on=['Season', 'Wteam'])
df_concat = pd.merge(left=df_dummy, right=df_lossseeds, on=['Season', 'Lteam'])
df_concat['seed_diff'] = df_concat.win_seed - df_concat.loss_seed

# positive seed_diff means the underdog won
# negative seed_diff means favorites won

# Make a new df with just wins and losses
df_wins = pd.DataFrame()
df_wins['seed_diff'] = df_concat['seed_diff']
df_wins['result'] = 1

df_losses = pd.DataFrame()
df_losses['seed_diff'] = -df_concat['seed_diff']
df_losses['result'] = 0

df_for_predictions = pd.concat((df_wins, df_losses))

# Create our training dataset
X_train = df_for_predictions.seed_diff.values.reshape(-1,1)
y_train = df_for_predictions.result.values
X_train, y_train = shuffle(X_train, y_train)

# Create our linear regression model using training data
model = LogisticRegression()
model = model.fit(X_train, y_train)

# score the model
print(model.score(X_train, y_train))

# examine classifier predictions
X = np.arange(-16, 16).reshape(-1, 1)
preds = model.predict_proba(X)[:,1]

# plot classifier predictions
plt.plot(X, preds)
plt.xlabel('Team1 seed - Team2 seed')
plt.ylabel('P(Team1 will win)')

# incorporate test data
df_sample_sub = pd.read_csv('Datasets/SampleSubmission.csv')
n_test_games = len(df_sample_sub)

def get_year_t1_t2(id):
    """Return a tuple with ints `year`, `team1` and `team2`."""
    return (int(x) for x in id.split('_'))
    
"""
We loop over each row in the sample_submission.csv file. 
For each row, we extract the year and the teams playing. 
We then look up the seeds for each of those teams in that season. 
Finally we add the seed difference to an array.
"""
X_test = np.zeros(shape=(n_test_games, 1))
for ii, row in df_sample_sub.iterrows():
    year, t1, t2 = get_year_t1_t2(row.Id)
    # There absolutely must be a better way of doing this!
    t1_seed = df_seeds[(df_seeds.Team == t1) & (df_seeds.Season == year)].n_seed.values[0]
    t2_seed = df_seeds[(df_seeds.Team == t2) & (df_seeds.Season == year)].n_seed.values[0]

    diff_seed = t2_seed - t1_seed
    X_test[ii, 0] = diff_seed

print(X_test)

# run trained model on test data
preds = model.predict_proba(X_test)

clipped_preds = np.clip(preds, 0.05, 0.95)
df_sample_sub.Pred = clipped_preds

# write to csv with results
df_sample_sub.to_csv('Datasets/submission.csv')

# TODO: Try incorporating ELO
