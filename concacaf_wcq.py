#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 20:24:24 2021

@author: thomasgaudin
"""
import csv, random
import numpy as np
import pandas as pd
from copy import deepcopy

################# FUNCTIONS ################################

def calculate_We(Ro, opponent_Ro, location):
    """ Calculate the We from the formula given by ELO """
    if location == 'home':
        dr = (Ro + 100) - opponent_Ro
        
    elif location == 'away':
        dr = Ro - (opponent_Ro + 100)
    
    We = 1 / (10 ** (-dr / 400) + 1)
    
    return We

def davidson_home_wp(home_We, away_We, theta):
    
    hwp = home_We / (home_We + (theta * away_We) )
    
    return hwp

def davidson_away_wp(home_We, away_We, theta):
    
    awp = away_We / ( (theta * home_We) + away_We)
    
    return awp

def davidson_tie_prob(home_We, away_We, theta):
    
    tie = ( (theta**2 - 1) * home_We * away_We ) / ((home_We + (theta * away_We) ) * ( (theta * home_We) + away_We))
    
    return tie

def calculate_home_win_probability(home_Ro, away_Ro):
    """ Win probability formula from online """
    
    wp = min((1 / (1 + 10**((away_Ro - home_Ro)/400)))**1.75 + 0.1, 1)
    
    return wp

def calculate_away_win_probability(home_Ro, away_Ro):
    """ Win probability formula from online """
    
    wp = max((1 / (1 + 10**((home_Ro - away_Ro)/400)))**1.75 - 0.1, 0)
    
    return wp

def calculate_elo(Ro, opponent_Ro, We, WLD, GD):
    """ ELO formula used for calculation """
    
    if GD < 2:
        GDM = 1
    
    elif GD == 2:
        GDM = GD * 1.5
        
    elif GD == 3:
        GDM == GD * 1.75
        
    elif GD >= 4:
        GDM = GD * (1.75 + (GD - 3) / 8 )
    
    K = 40 * GDM
    
    Rn = Ro + K * (WLD - We)
    
    return Rn

#Initialize files

schedule_file = 'concacaf_schedule.csv'

ranking_file = 'concacaf_elo.csv'

table_file = 'concacaf_table.csv'

#initialize lists, dictionaries, arrays

matches = []

init_elo_rank = dict()

elo_rank = dict()

init_oct_table = dict()

oct_table = dict()

final_table = dict()

total_table = dict()

average_final_table = dict()

#read schedule and append to list
with open(schedule_file, 'r') as sched_file:
    read_schedule = csv.reader(sched_file)
    
    location = next(read_schedule)
    
    for match in read_schedule:
        vals = match[0].split(', ')
        #print(vals)
        matches.append(vals)
        
#read elo rankings and append to dicts
with open(ranking_file, 'r') as elo_file:
    read_elo = csv.reader(elo_file)
    
    elo_header = next(read_elo)
    
    for team in read_elo:
        #print(team)
        elo_rank[team[0]] = int(team[1])
        init_elo_rank[team[0]] = int(team[1])

#read table and append to dicts
with open(table_file, 'r') as tab_file:
    read_table = csv.reader(tab_file)
    
    table_header = next(read_table)
    
    for row in read_table:
        oct_table[row[0]] = int(row[1])
        init_oct_table[row[0]] = int(row[1])
        
#list teams and count them        
list_teams = []
for key in oct_table.keys():
    list_teams.append(key)
    
num_teams = len(list_teams)

#pandas data frame for table
points_table = pd.DataFrame(oct_table.items(), 
                               index = range(1, num_teams+1),
                               columns = ['Team', 'Points'])

init_points_table = pd.DataFrame(init_oct_table.items(), 
                               index = range(1, num_teams+1),
                               columns = ['Team', 'Points'])

#initialize table for counting how often teams end with a certain rank
placement_table = pd.DataFrame(0, 
                               index = np.arange(1, num_teams+1), 
                               columns = oct_table.keys())
#print(points_table)
#print(placement_table)


#generate the blank total points table
for team in oct_table.keys():
    total_table[team] = []
    
#print(elo_rank)

#print(oct_table)

#print(matches[0:4])

iterations = 10000

count = 1

theta = 1.7 #hyperparameter for Davidson functions

for num in range(iterations):
    
    for match in matches:
        
        #initialize home team and ELO
        home_team = match[0]
        home_elo = elo_rank[home_team]
        
        #initialize away team and ELO
        away_team = match[1]
        away_elo = elo_rank[away_team]
        
        #calculate We for new ELO calc
        home_we = calculate_We(home_elo, away_elo, 'home')
        away_we = calculate_We(away_elo, home_elo, 'away')
        
        #Determine win probability for each team
        #home_wp = calculate_home_win_probability(home_elo, away_elo)
        home_wp = davidson_home_wp(home_we, away_we, theta)
        #away_wp = calculate_away_win_probability(home_elo, away_elo)
        away_wp = davidson_away_wp(home_we, away_we, theta)

        #draw_wp = 1 - home_wp - away_wp
        draw_wp = davidson_tie_prob(home_we, away_we, theta)
        
        #print(f'{home_team} / draw / {away_team}')
        #print(f'{round(home_wp,2)} / {round(draw_wp,2)} / {round(away_wp,2)}')
        
        #sort weights, outcomes dict: win = 1, draw = 0.5, loss = 0.0
        weights = {1.0: home_wp, 0.5: draw_wp, 0.0: away_wp}
        sorted_weights = {k: v for k, v in sorted(weights.items(), key=lambda item: item[1])}
        
        #print(sorted_weights)
        
        weights_list = []
        
        outcomes = []
        probabilities = []
        
        for weight in sorted_weights.keys():
            weights_list.append((weight, sorted_weights[weight]))
        
        for outcome in weights_list:
            outcomes.append(outcome[0])
            
        for probability in weights_list:
            probabilities.append(probability[1])
        
        #choose a random outcome
        outcome = random.choices(outcomes, weights=probabilities, k=1)
        
        #print(outcomes)
        #print(probabilities)
        #print(outcome)
        
        #home win
        if outcome[0] == 1:
            
            #update table
            points_table.loc[points_table['Team'] == home_team, ['Points']] += 3
            points_table.loc[points_table['Team'] == away_team, ['Points']] += 0
            
            #new home elo
            new_home_elo = calculate_elo(home_elo, away_elo, home_we, outcome[0], 1)
            
            elo_rank[home_team] = new_home_elo
            
            #new away elo
            new_away_elo = calculate_elo(away_elo, home_elo, away_we, 0, 1)
            
            elo_rank[away_team] = new_away_elo
            
        elif outcome[0] == 0.5:
            
            #update table
            points_table.loc[points_table['Team'] == home_team, ['Points']] += 1
            points_table.loc[points_table['Team'] == away_team, ['Points']] += 1
            
            #new home elo
            new_home_elo = calculate_elo(home_elo, away_elo, home_we, outcome[0], 0)
            
            elo_rank[home_team] = new_home_elo
            
            #new away elo
            new_away_elo = calculate_elo(away_elo, home_elo, away_we, outcome[0], 0)
            
            elo_rank[away_team] = new_away_elo
            
        #away win    
        else:
        
            #update table
            points_table.loc[points_table['Team'] == home_team, ['Points']] += 0
            points_table.loc[points_table['Team'] == away_team, ['Points']] += 3
            
            #new home elo
            new_home_elo = calculate_elo(home_elo, away_elo, home_we, 0, 1)
            
            elo_rank[home_team] = new_home_elo
            
            #new away elo
            new_away_elo = calculate_elo(away_elo, home_elo, away_we, outcome[0], 1)
            
            elo_rank[away_team] = new_away_elo
        

    #create final table, append to total table, and reset to initial table
    final_table = deepcopy(points_table)
    
    for team in list_teams:
        final_points = final_table.loc[final_table['Team'] == team, 'Points'].values[0]
        total_table[team].append(final_points)
    
    points_table = deepcopy(init_points_table)
    
    #reset ELO rankings
    for team in elo_rank.keys():
        elo_rank[team] = deepcopy(init_elo_rank[team])
    
    #rank final table and append to placement table
    final_table['Rank'] = final_table['Points'].rank(ascending = False)
    for team in list_teams:
        rank = int(final_table.loc[final_table['Team'] == team, ['Rank']].values[0])
        placement_table.loc[rank, team] += 1
        
    print(count)
    count += 1
        
    #print(final_table)
    #print(elo_rank)
    
    #print(final_table)

#create average points gained table    
for team in list_teams:
    total_points = sum(total_table[team])
        
    average_final_table[team] = round(total_points / iterations, 1)

avg_final_table = pd.DataFrame(average_final_table.items(), 
                               index = range(1, num_teams+1),
                               columns = ['Team', 'Points'])

final_tab = avg_final_table.sort_values('Points', ascending = False)

#create percent chance to finish in each spot table
percent_finish = (1 / iterations) * placement_table 

print(percent_finish)

#output csv of percent finish

#calculate percent qualify and append to average points table
final_tab['Chance to Qualify'] = 0

for column in percent_finish:
    percent_qualify = sum(percent_finish.loc[0:3, column].values)
    
    final_tab.loc[final_tab['Team'] == column, ['Chance to Qualify']] = percent_qualify
    

    
    
print(final_tab)