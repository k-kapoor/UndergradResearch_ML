'''
  File name: dataexploration.py
  Author: Kunal Kapoor
  Date Created: 4/5/2021
  Date last modified: 4/5/2021
'''

#IMPORTS
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as datez
import datetime


#HELPER FUNCTIONS
def getMaleCount(df):
 return df['author_ID'][df['Gender'] == 'M'].nunique()
def getFemaleCount(df):
  return df['author_ID'][df['Gender'] == 'F'].nunique()
def getMaleCommits(df):
  return df[df['Gender'] == 'M'].shape[0]
def getFemaleCommits(df):
  return df[df['Gender'] == 'F'].shape[0]
   
#DATA EXPLORATION FUNCTIONS
def messageLen(df):
  mCommits = getMaleCommits(df)
  fCommits = getFemaleCommits(df)
  m_len = df['MSG_LEN'][df['Gender'] == 'F'].sum()
  f_len = df['MSG_LEN'][df['Gender'] == 'M'].sum()
  print('Average male message length: ', m_len/mCommits)
  print('Average female message length: ', f_len/fCommits, '\n')
def printDemographics(df):
  #print('Number of Teams: ', getMaleCount(df))
  print('Average number of commits per male: ', getMaleCommits(df)/getMaleCount(df))
  print('Average number of commits per female: ', getFemaleCommits(df)/getFemaleCount(df), '\n')
def getTagBreakup(df):
  list = df['Tag'][df['Gender'] == 'M'].value_counts()
  print('Percentage of Commits by Tag (Male): \n', list/getMaleCommits(df))
  list = df['Tag'][df['Gender'] == 'F'].value_counts()
  print('Percentage of Commits by Tag (Female): \n', list/getFemaleCommits(df), '\n')
  
#VISUALIZATION FUNCTIONS  
#plot graph of commits by dates per gender
def plotDateTimeByGender(pre_process):
  pre_process['ID'] = pre_process.index
  pre_process['human_time'] = pd.to_datetime(pre_process['Commit_datetime'], format = '%Y-%m-%d %H:%M:%S').dt.date
  commits_by_gender= pre_process.groupby(['human_time', 'Gender'], as_index=False)['ID'].count()
  group_author = pre_process.groupby(['human_time', 'Gender'])['author_ID'].nunique().reset_index()
  plot_data = pd.concat([commits_by_gender, group_author], axis=1)
  plot_data['mean'] = plot_data.ID/plot_data.author_ID
  plot_data = plot_data.loc[:, ~plot_data.columns.duplicated()]
  maleDateCommits = plot_data['mean'][plot_data['Gender'] == 'M']
  femaleDateCommits = plot_data['mean'][plot_data['Gender'] == 'F']
  maleDates = plot_data['human_time'][plot_data['Gender'] == 'M']
  femaleDates = plot_data['human_time'][plot_data['Gender'] == 'F']
  plt.plot(maleDates, maleDateCommits, "-r", label = "Males")
  plt.plot(femaleDates, femaleDateCommits, "-b", label = "Females")
  plt.title('Commits by Time')
  plt.ylabel('Commits per Contributor')
  plt.xlabel('Date')
  plt.legend(loc = "upper left")
  plt.show()

def plotGenderCommits(gender, pre_process):
  (pre_process['Gender'].value_counts() / gender['Gender'].value_counts()).plot(kind='bar', width = .6, color=['red', 'blue', 'green'])
  plt.xlabel("Gender")
  plt.ylabel("Average # of commits made")
  plt.title("Average # of commits made by Gender")
  plt.show()

pre_process = pd.read_csv('../csv/pre_processed.csv')
mixedTeamsDF = pd.read_csv('../csv/mixedTeams.csv')
gender = pd.read_csv('../csv/Gender.csv')
gender.fillna('U.S', inplace = True)

#MIXED TEAM DATA ANALYSIS
messageLen(mixedTeamsDF)
printDemographics(mixedTeamsDF)
getTagBreakup(mixedTeamsDF)

#NORMAL ANALYSIS
'''messageLen(pre_process)
printDemographics(pre_process)
getTagBreakup(pre_process)'''

#BOTH ANALYSIS
plotDateTimeByGender(pre_process)
plotGenderCommits(gender, pre_process)



#gender analysis
'''print(pre_process['Gender'].value_counts())
print(pre_process['author_ID'][pre_process['Gender'] == 'U.S'].nunique())
print(pre_process['author_ID'][pre_process['Gender'] == 'U.S'].value_counts())
print(pre_process['Added'][pre_process['Gender'] == 'M'].sum())
print(pre_process['Added'][pre_process['Gender'] == 'U.S'].sum())
print(pre_process['Added'][pre_process['Gender'] == 'F'].sum())
print('break')
print(pre_process['Deleted'][pre_process['Gender'] == 'M'].sum())
print(pre_process['Deleted'][pre_process['Gender'] == 'U.S'].sum())
print(pre_process['Deleted'][pre_process['Gender'] == 'F'].sum())
print('break')
print(pre_process['Modified'][pre_process['Gender'] == 'M'].sum())
print(pre_process['Modified'][pre_process['Gender'] == 'U.S'].sum())
print(pre_process['Modified'][pre_process['Gender'] == 'F'].sum())
print('break')
print(pre_process['Copied'][pre_process['Gender'] == 'M'].sum())
print(pre_process['Copied'][pre_process['Gender'] == 'U.S'].sum())
print(pre_process['Copied'][pre_process['Gender'] == 'F'].sum())
print('break')
print(pre_process['Removed'][pre_process['Gender'] == 'M'].sum())
print(pre_process['Removed'][pre_process['Gender'] == 'U.S'].sum())
print(pre_process['Removed'][pre_process['Gender'] == 'F'].sum())

#tags
print(pre_process['Tag'].nunique())
print(pre_process['Gender'][pre_process['Tag'] == 'Untagged'].value_counts())
print(pre_process['Project_Type'][pre_process['Tag'] == 'Untagged'].value_counts())
print(pre_process['Project_Type'].value_counts())

#max partners within team p2
maxPartners = 0
for i in range(10,73):
  partners = pre_process['author_ID'][pre_process['Repo_name'] == repo + str(i)].nunique()
  if partners > maxPartners:
    maxPartners = partners
print(maxPartners)

#mixedTeams analysis
print(mixedTeamsDF.shape[0])
print(mixedTeamsDF[mixedTeamsDF['Gender'] == 'F']['Commit_Type'].value_counts())
print(mixedTeamsDF[mixedTeamsDF['Gender'] == 'M']['Commit_Type'].value_counts())
print(mixedTeamsDF[mixedTeamsDF['Gender'] == 'F']['Tag'].value_counts())
print(mixedTeamsDF[mixedTeamsDF['Gender'] == 'M']['Tag'].value_counts())
print(mixedTeamsDF[mixedTeamsDF['Gender'] == 'M']['Commit_message'].str.len)
print(mixedTeamsDF[mixedTeamsDF['Gender'] == 'F']['Commit_message'].str.len)'''
'''print("MaleT: " + str(maleTeams) + "  FemaleT: " + str(femaleTeams) + "  MixedT: " + 
  str(mixedTeams))
print("MaleCommits: " + str(maleCommits) + "  FemaleCommits: " + str(femaleCommits) + 
  "  MixedCommits: " + str(mixedCommits))
print("MixedMaleCommits: " + str(mixedMCommits) + "  MixedFemaleCommits: " + str(mixedFCommits))
print("MixedMaleLinesAdded: " + str(mixedMLinesAdded) + "  MixedFemaleLinesAdded: " 
  + str(mixedFLinesAdded) + "  MaleLinesAdded: " + str(maleLinesAdded) + "  FemaleLinesAdded: " + 
  str(femaleLinesAdded))
print("MixedMaleLinesModified: " + str(mixedMLinesModified) + "  MixedFemaleLinesModified: " 
  + str(mixedFLinesModified) + "  MaleLinesModified: " + str(maleLinesModified) + 
  "  FemaleLinesModified: " + str(femaleLinesModified))'''
'''
#data analysis of day of week commits and length of messages
print(mixedTeamsDF['Day'][mixedTeamsDF['Gender'] == 'M'].value_counts())
print(mixedTeamsDF['Day'][mixedTeamsDF['Gender'] == 'F'].value_counts())
print(mixedTeamsDF['MSG_LEN'][mixedTeamsDF['Gender'] == 'F'].sum())
print(mixedTeamsDF['MSG_LEN'][mixedTeamsDF['Gender'] == 'M'].sum())'''