#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 17 19:54:56 2025

@author: melissafeeney
"""

# -------------------------
# IV. GENERATE MATCHUP FILE
# -------------------------

import pandas as pd
import itertools

# Generate file with all possible matchups
path = '/Users/melissafeeney/Desktop/marchmadness_2025/kaggle_data_31725/'
regseason = pd.read_csv(path + 'MRegularSeasonDetailedResults.csv')
teams = pd.read_csv(path + 'MTeams.csv')

# All the teams who are D1 in 2025 season
teams_2025 = teams[teams['LastD1Season'] == 2025]['TeamID'].values

matchups = itertools.combinations(teams_2025, 2)

matchups_list = []
for matchup in matchups:
    #print(matchup)
    matchups_list.append(matchup)
    
sample_submission = pd.DataFrame(matchups_list)
sample_submission.columns = ['team1', 'team2']
sample_submission['season'] = 2025

sample_submission['ID'] = sample_submission['season'].astype(str) + '_' + sample_submission['team1'].astype(str) + '_' + sample_submission['team2'].astype(str)
sample_submission['Pred'] = 0.5

del sample_submission['season']
del sample_submission['team1']
del sample_submission['team2']

sample_submission.to_csv('SampleSubmission2025.csv')