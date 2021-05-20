#!/usr/bin/env python
# coding: utf-8

# # Task - 5 Exploratory Data Analysis - Sports 

# Coder - MAHIPAL PAREEK

# Task- Perform ‘Exploratory Data Analysis’ on dataset ‘Indian Premier League’ 

# As a sports analysts, find out the most successful teams, players and factors 
# contributing win or loss of a team. 

# Suggest teams or players a company should endorse for its products. 
# 

# Dataset: https://bit.ly/34SRn3b

# Task completed during Data Science & Analytics Internship @ The Sparks Foundation

# ### Importing LIBRARIES:

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Loading 1st Dataset

# In[4]:


matches = pd.read_csv("C://Users//shubh//Downloads//matches.csv")


# In[5]:


matches.head()


# In[6]:


matches.tail()


# ### Data information:

# In[7]:


matches.info()


# In[8]:


matches.shape


# In[9]:


matches.describe()


# ### Loading 2nd Dataset

# In[11]:


deliveries = pd.read_csv("C://Users//shubh//Downloads//deliveries.csv")


# In[12]:


deliveries.head()


# In[13]:


deliveries.tail()


# ### Data information:

# In[14]:


deliveries.info()


# In[15]:


deliveries.shape


# In[16]:


deliveries.describe()


# ### Now, We will merge the 2 datasets for better insights from the data

# In[17]:


#merging the 2 datasets
merge = pd.merge(deliveries,matches, left_on='match_id', right_on ='id')
merge.head(2)


# In[18]:


merge.info()


# In[19]:


merge.describe()


# In[20]:


matches.id.is_unique


# In[21]:


matches.set_index('id', inplace=True)


# In[22]:



#Summary statistics of matches data
matches.describe(include = 'all')


# ### Data Preprocessing

# Here we will perform Data Preprocessing on our matches dataset first, to make the data usable for EDA.

# In[23]:


matches.head()


# ### From Pre profiling, we found that:

# city has missing values
# 
# team1 and team2 columns have 14 distinct values but winner has 15 distinct values
# 
# umpire1 and umpire2 have 1 missing value each
# 
# umpire3 has 94% missing values
# 
# city has 33 distinct values while venue has 35 distinct values

# ### Filling in the missing values of city column

# ### First let's find the venues corresponding to which the values of city are empty

# In[25]:


matches[matches.city.isnull()][['city','venue']]


# So, missing values can be filled with Dubai

# In[26]:


matches.city = matches.city.fillna('Dubai')


# umpire1 and umpire2 columns have one missing value each.

# In[27]:


matches[(matches.umpire1.isnull()) | (matches.umpire2.isnull())]


# Umpire3 column has close to 92% missing values. hence dropping that column

# In[28]:


matches = matches.drop('umpire3', axis = 1)


# In[29]:


#city has 33 distinct values while we have 35 venues.
#Let's find out venues grouped by cities to see which cities have multiple venues

city_venue = matches.groupby(['city','venue']).count()['season']
city_venue_df = pd.DataFrame(city_venue)
city_venue_df


# ## Observations

# Bengaluru and Bangalore both are in the data when they are same. So we need to keep one of them. 
# 
# Chandigarh and Mohali are same and there is just one stadium Punjab Cricket Association IS Bindra Stadium, Mohali whose value has not been entered correctly. We need to have either Chandigarh or Mohali as well as correct name of the stadium there. 
# 
# Mumbai has 3 stadiums/venues used for IPL. 
# 
# Pune has 2 venues for IPL. 
# 
# ### Visual representation of number of venues in each city .

# In[30]:


#Plotting venues along with cities 
v = pd.crosstab(matches['city'],matches['venue'])
v.replace(v[v!=0],1, inplace = True)

#Adding a column by summing each columns
v['count'] = v.sum(axis = 'columns')
#We will just keep last column = 'count'
b = v['count']

#Plotting
plt.figure(figsize = (20,7))
b.plot(kind = 'bar')
plt.title("Number of stadiums in different cities", fontsize = 25, fontweight = 'bold')
plt.xlabel("City", size = 30)
plt.ylabel("Frequency", size = 30)
plt.xticks(size = 20)
plt.yticks(size = 20)


# ### Exploratory Data Analysis:

# ### Number of matches played in each season

# In[31]:


plt.figure(figsize=(15,5))
sns.countplot('season', data = matches)
plt.title("Number of matches played each season",fontsize=18,fontweight="bold")
plt.ylabel("Count", size = 25)
plt.xlabel("Season", size = 25)
plt.xticks(size = 20)
plt.yticks(size = 20)


# 2011-2013 have more matches being played than other seasons
# 
# All other seasons have approximately 58-60 matches while 2011-2013 have more than 70 matches.

# ### How many teams played in each season?

# In[32]:


matches.groupby('season')['team1'].nunique().plot(kind = 'bar', figsize=(15,5))
plt.title("Number of teams participated each season ",fontsize=18,fontweight="bold")
plt.ylabel("Count of teams", size = 25)
plt.xlabel("Season", size = 25)
plt.xticks(size = 15)
plt.yticks(size = 15)


# 10 teams played in 2011 and 9 teams each in 2012 and 2013
# 
# This explains why 2011-2013 have seen more matches being played than other seasons

# ### Venue which has hosted most number of IPL matches .

# In[33]:


matches.venue.value_counts().sort_values(ascending = True).tail(10).plot(kind = 'barh',figsize=(12,8), fontsize=15, color='green')
plt.title("Venue which has hosted most number of IPL matches",fontsize=18,fontweight="bold")
plt.ylabel("Venue", size = 25)
plt.xlabel("Frequency", size = 25)


# M Chinnaswamy Stadium in Bengaluru has hosted the highest number of matches so far in IPL followed by Eden Gardens in Kolkata

# ### Which team has maximum wins in IPL so far?

# In[34]:


#creating a dataframe with season and winner columns
winning_teams = matches[['season','winner']]


# In[35]:


#dictionaries to get winners to each season
winners_team = {}
for i in sorted(winning_teams.season.unique()):
    winners_team[i] = winning_teams[winning_teams.season == i]['winner'].tail(1).values[0]
    
winners_of_IPL = pd.Series(winners_team)
winners_of_IPL = pd.DataFrame(winners_of_IPL, columns=['team'])


# In[36]:


winners_of_IPL['team'].value_counts().plot(kind = 'barh', figsize = (15,5), color = 'darkblue')
plt.title("Winners of IPL across 11 seasons",fontsize=18,fontweight="bold")
plt.ylabel("Teams", size = 25)
plt.xlabel("Frequency", size = 25)
plt.xticks(size = 15)
plt.yticks(size = 15)


# MI and CSK have both won 3 times each followed by KKR who has won 2 times.
# Hyderabad team has also won 2 matches under 2 franchise name - Deccan Chargers and Sunrisers Hyderabad

# ### Does teams choosed to bat or field first, after winning toss?

# In[37]:


matches['toss_decision'].value_counts().plot(kind='pie', fontsize=14, autopct='%3.1f%%', 
                                               figsize=(10,7), shadow=True, startangle=135, legend=True, cmap='Oranges')

plt.ylabel('Toss Decision')
plt.title('Decision taken by captains after winning tosses')


# Close to 60% times teams who have won tosses have decided to chase down

# ### How toss decision affects match results?

# In[38]:


matches['toss_win_game_win'] = np.where((matches.toss_winner == matches.winner),'Yes','No')
plt.figure(figsize = (15,5))
sns.countplot('toss_win_game_win', data=matches, hue = 'toss_decision')
plt.title("How Toss Decision affects match result", fontsize=18,fontweight="bold")
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.xlabel("Winning Toss and winning match", fontsize = 25)
plt.ylabel("Frequency", fontsize = 25)


# Teams winning tosses and electng to field first have won most number of times.

# ### Individual teams decision to choose bat or field after winning toss.

# In[39]:


plt.figure(figsize = (25,10))
sns.countplot('toss_winner', data = matches, hue = 'toss_decision')
plt.title("Teams decision to bat first or second after winning toss", size = 30, fontweight = 'bold')
plt.xticks(size = 10)
plt.yticks(size = 15)
plt.xlabel("Toss Winner", size = 35)
plt.ylabel("Count", size = 35)


# Most teams field first after winning toss except for Chennai Super Kings who has mostly opted to bat first. Deccan Chargers and Pune Warriors also show the same trend.

# ### Which player's performance has mostly led team's win?

# In[40]:


MoM= matches['player_of_match'].value_counts()
MoM.head(10).plot(kind = 'bar',figsize=(12,8), fontsize=15, color='black')
plt.title("Top 10 players with most MoM awards",fontsize=18,fontweight="bold")
plt.ylabel("Frequency", size = 25)
plt.xlabel("Players", size = 25)


# Chris Gayle has so far won the most number of MoM awards followed by AB de Villiers.
# Also, all top 10 are batsmen which kind of hints that in IPL batsmen have mostly dictated the matches

# ### How winning matches by fielding first varies across venues?

# In[41]:


new_matches = matches[matches['result'] == 'normal']   #taking all those matches where result is normal and creating a new dataframe
new_matches['win_batting_first'] = np.where((new_matches.win_by_runs > 0), 'Yes', 'No')
new_matches.groupby('venue')['win_batting_first'].value_counts().unstack().plot(kind = 'barh', stacked = True,
                                                                               figsize=(15,15))
plt.title("How winning matches by fielding first varies across venues?", fontsize=18,fontweight="bold")
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.xlabel("Frequency", fontsize = 25)
plt.ylabel("Venue", fontsize = 25)


# Batting second has been more rewarding in almost all the venues

# ### Is batting second advantageous across all years?

# In[42]:


plt.figure(figsize = (15,5))
sns.countplot('season', data = new_matches, hue = 'win_batting_first')
plt.title("Is batting second advantageous across all years", fontsize=20,fontweight="bold")
plt.xticks(size = 15)
plt.yticks(size = 15)
plt.xlabel("Season", fontsize = 25)
plt.ylabel("Count", fontsize = 25)


# Exceptt for 2010 and 2015, in all other years it can be seen that teams batting second have won more matches

# ### Teams total scoring runs, over the years?

# In[43]:


merge.groupby('season')['batsman_runs'].sum().plot(kind = 'line', linewidth = 3, figsize =(15,5),
                                                                                             color = 'blue')
plt.title("Runs over the years",fontsize= 25, fontweight = 'bold')
plt.xlabel("Season", size = 25)
plt.ylabel("Total Runs Scored", size = 25)
plt.xticks(size = 12)
plt.yticks(size = 12)


# Run scoring has gone up from the start of the IPL in 2008.

# ### Top Run Getters of IPL.

# In[44]:


#let's plot the top 10 run getter so far in IPL
merge.groupby('batsman')['batsman_runs'].sum().sort_values(ascending = False).head(10).plot(kind = 'bar', color = 'red',
                                                                                            figsize = (15,5))
plt.title("Top Run Getters of IPL", fontsize = 20, fontweight = 'bold')
plt.xlabel("Batsmen", size = 25)
plt.ylabel("Total Runs Scored", size = 25)
plt.xticks(size = 12)
plt.yticks(size = 12)


# Except for MS Dhoni, all other top run getters are either openers or come in 3rd or 4th positions to bat.
# 
# Suresh Raina is the highest run getter in IPL.

# ### Which batsman has been most consistent among top 10 run getters?

# In[45]:


consistent_batsman = merge[merge.batsman.isin(['SK Raina', 'V Kohli','RG Sharma','G Gambhir',
                                            'RV Uthappa', 'S Dhawan','CH Gayle', 'MS Dhoni',
                                            'DA Warner', 'AB de Villiers'])][['batsman','season','total_runs']]

consistent_batsman.groupby(['season','batsman'])['total_runs'].sum().unstack().plot(kind = 'box', figsize = (15,8))
plt.title("Most Consistent batsmen of IPL", fontsize = 20, fontweight = 'bold')
plt.xlabel("Batsmen", size = 25)
plt.ylabel("Total Runs Scored each season", size = 25)
plt.xticks(size = 15)
plt.yticks(size = 15)


# Median score for Raina is above all the top 10 run getters. He has the highest lowest run among all the batsmen across 11 seasons. Considering the highest and lowest season totals and spread of runs, it seems Raina has been most consistent among all.

# ### Which bowlers have performed the best?

# In[47]:


merge.groupby('bowler')['player_dismissed'].count().sort_values(ascending = False).head(10).plot(kind = 'bar', 
                                                color = 'purple', figsize = (15,5))
plt.title("Top Wicket Takers of IPL", fontsize = 20, fontweight = 'bold')
plt.xlabel("Bowler", size = 25)
plt.ylabel("Total Wickets Taken", size = 25)
plt.xticks(size = 12)
plt.yticks(size = 12)


# Malinga has taken the most number of wickets in IPL followed by Bravo and Amit Mishra
# 
# In top 10 bowlers, 5 are fast and medium pacers while the other 5 are spinners
# 
# All 5 spinners are right arm spinners and 2 are leg spinners while 3 are off spinners
# 
# All 5 pacers are right arm pacers

# ### Batsmen with the best strike rates over the years .

# In[49]:


#We will consider players who have played 10 or more seasons
no_of_balls = pd.DataFrame(merge.groupby('batsman')['ball'].count()) #total number of matches played by each batsman
runs = pd.DataFrame(merge.groupby('batsman')['batsman_runs'].sum()) #total runs of each batsman
seasons = pd.DataFrame(merge.groupby('batsman')['season'].nunique()) #season = 1 implies played only 1 season

batsman_strike_rate = pd.DataFrame({'balls':no_of_balls['ball'],'run':runs['batsman_runs'],'season':seasons['season']})
batsman_strike_rate.reset_index(inplace = True)

batsman_strike_rate['strike_rate'] = batsman_strike_rate['run']/batsman_strike_rate['balls']*100
highest_strike_rate = batsman_strike_rate[batsman_strike_rate.season.isin([10,11])][['season','batsman','strike_rate']].sort_values(by = 'strike_rate',
                                                                                                            ascending = False)

highest_strike_rate.head(10)


# In[50]:


plt.figure(figsize = (15,6))
sns.barplot(x='batsman', y='strike_rate', data = highest_strike_rate.head(10), hue = 'season')
plt.title("Highest strike rates in IPL",fontsize= 30, fontweight = 'bold')
plt.xlabel("Player", size = 25)
plt.ylabel("Strike Rate", size = 25)
plt.xticks(size = 14)
plt.yticks(size = 14)


# AB de Villiers, Gayle have the highest strike rates in IPL. They are the big hitters and can win any match on their day
# 
# One surprise here is that Harbhajan Singh who is a bowler has a strike rate of 130+ and comes before Rohit Sharma in ranking

# ### Bowlers with maximum number of extras.

# In[51]:


extra = deliveries[deliveries['extra_runs']!=0]['bowler'].value_counts()[:10]
extra.plot(kind='bar', figsize=(11,6), title='Bowlers who have bowled maximum number of Extra balls')

plt.xlabel('BOWLER')
plt.ylabel('BALLS')
plt.show()

extra = pd.DataFrame(extra)
extra.T


# ### Which bowlers have picked up wickets more frequently?

# In[52]:


#strike_rate = balls bowled by wickets taken
balls_bowled = pd.DataFrame(merge.groupby('bowler')['ball'].count())
wickets_taken = pd.DataFrame(merge[merge['dismissal_kind'] != 'no dismissal'].groupby('bowler')['dismissal_kind'].count())
seasons_played = pd.DataFrame(merge.groupby('bowler')['season'].nunique())
bowler_strike_rate = pd.DataFrame({'balls':balls_bowled['ball'],'wickets':wickets_taken['dismissal_kind'],
                          'season':seasons_played['season']})
bowler_strike_rate.reset_index(inplace = True)


# In[53]:


bowler_strike_rate['strike_rate'] = bowler_strike_rate['balls']/bowler_strike_rate['wickets']
def highlight_cols(s):
    color = 'skyblue'
    return 'background-color: %s' % color
#Strike rate for bowlers who have taken more than 50 wickets
best_bowling_strike_rate = bowler_strike_rate[bowler_strike_rate['wickets'] > 50].sort_values(by = 'strike_rate', ascending = True)
best_bowling_strike_rate.head().style.applymap(highlight_cols, subset=pd.IndexSlice[:, ['bowler', 'wickets','strike_rate']])


# ## Q1. As a sports analysts, The most successful teams, players & factors contributing win or loss of a team:

# Mumbai Indians is the most successful team in IPL and has won the most number of toss.
# 
# There were more matches won by chasing the total(419 matches) than defending(350 matches).
# 
# When defending a total, the biggest victory was by 146 runs(Mumbai Indians defeated Delhi Daredevils by 146 runs on 06 May 2017 at Feroz Shah Kotla stadium, Delhi).
# 
# When chasing a target, the biggest victory was by 10 wickets(without losing any wickets) and there were 11 such instances.
# 
# The Mumbai city has hosted the most number of IPL matches.
# 
# Chris Gayle has won the maximum number of player of the match title.
# 
# S. Ravi(Sundaram Ravi) has officiated the most number of IPL matches on-field.
# 
# Eden Gardens has hosted the maximum number of IPL matches.
# 
# If a team wins a toss choose to field first as it has highest probablity of winning

# ## Q2. Teams or Players a company should endorse for its products.

# If the franchise is looking for a consistant batsman who needs to score good amount of runs then go for V Kohli, S Raina, Rohit Sharma , David Warner...
# 
# If the franchise is looking for a game changing batsman then go for Chris Gayle, AB deVillers, R Sharma , MS Dhoni...
# 
# If the franchise is looking for a batsman who could score good amount of runs every match the go for DA Warner, CH Gayle, V Kohli,AB de Villiers,S Dhawan
# 
# If the franchise needs the best finisher in lower order having good strike rate then go for CH Gayle,KA Pollard, DA Warner,SR Watson,BB McCullum
# 
# If the franchise need a experienced bowler then go for Harbhajan Singh ,A Mishra,PP Chawla ,R Ashwin,SL Malinga,DJ Bravo
# 
# If the franchise need a wicket taking bowler then go for SL Malinga,DJ Bravo,A Mishra ,Harbhajan Singh, PP Chawla
# 
# If the franchise need a bowler bowling most number of dot balls then go for Harbhajan Singh,SL Malinga,B Kumar,A Mishra,PP Chawla
# 
# If the franchise need a bowler with good economy then go for DW Steyn ,M Muralitharan ,R Ashwin,SP Narine ,Harbhajan Singh

# # ---------------------------------- End of Code ---------------------------------
