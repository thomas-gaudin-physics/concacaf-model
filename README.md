Welcome to the site of Tom's Model for predictive modeling of CONCACAF World Cup Qualifying. 
This model uses Elo ratings from eloratings.net and a few formulae found in papers to predict the results of the region's qualifiying tournament.

Use is simple. Create three CSV files of the following format:

Elo file:
Heading: "Team", "ELO Rating"
team, rating

Schedule file:
Heading: "home", "away"
home team, away team

Table file:
Heading: "Team", "Points"
team, current points

The code will do everything else for you, returning a finishing place table and average final table
