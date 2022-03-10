'''
  File name: preprocess.py
  Author: Kunal Kapoor
  Date Created: 4/6/2021
  Date last modified: 4/6/2021
'''

import pandas as pd
import re
import enchant
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from sklearn.model_selection import train_test_split
import datetime
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

#split data by repo
def splitByRepo(pre_process):
  pre_process.loc[pre_process['Repo_name'].str.contains('csc216-001-P1'), 
  'Project_Type'] = 'P1'
  pre_process.loc[pre_process['Repo_name'].str.contains('csc216-002-P1'), 
  'Project_Type'] = 'P1'
  pre_process.loc[pre_process['Repo_name'].str.contains('csc216-601-P1'), 
    'Project_Type'] = 'P1'
  pre_process.loc[pre_process['Repo_name'].str.contains('csc216-001-P2'), 
    'Project_Type'] = 'P2'
  pre_process.loc[pre_process['Repo_name'].str.contains('csc216-002-P2'), 
    'Project_Type'] = 'P2'
  pre_process.loc[pre_process['Repo_name'].str.contains('csc216-601-P2'), 
    'Project_Type'] = 'P2'
  pre_process.loc[pre_process['Repo_name'].str.contains('(?i)csc217'), 
    'Project_Type'] = 'Lab'
  pre_process.loc[pre_process['Repo_name'].str.contains('csc216-001-GP1'), 
    'Project_Type'] = 'GP1'
  pre_process.loc[pre_process['Repo_name'].str.contains('csc216-002-GP1'), 
    'Project_Type'] = 'GP1'
  pre_process.loc[pre_process['Repo_name'].str.contains('csc216-601-GP1'), 
    'Project_Type'] = 'GP1'
  pre_process.loc[pre_process['Repo_name'].str.contains('csc216-001-GP2'), 
    'Project_Type'] = 'GP2'
  pre_process.loc[pre_process['Repo_name'].str.contains('csc216-002-GP2'), 
    'Project_Type'] = 'GP2'
  pre_process.loc[pre_process['Repo_name'].str.contains('csc216-601-GP2'), 
    'Project_Type'] = 'GP2'
  pre_process.loc[pre_process['Repo_name'].str.contains('csc216-001-GP3'), 
    'Project_Type'] = 'GP3'
  pre_process.loc[pre_process['Repo_name'].str.contains('csc216-002-GP3'), 
    'Project_Type'] = 'GP3'
  pre_process.loc[pre_process['Repo_name'].str.contains('csc216-601-GP3'), 
    'Project_Type'] = 'GP3'
  pre_process.loc[pre_process['Repo_name'].str.contains('csc216-T'), 
    'Project_Type'] = 'Team P2'
  return pre_process

#classify tags and messages according to keywords
def classifyTagMessage(pre_process):
  #clean tag
  pre_process['Tag_type'] = ''
  pre_process.loc[pre_process['Tag'].str.contains('document|javadoc', case = False, na=False),
    'Tag_type'] = 'Documentation'
  pre_process.loc[pre_process['Tag'].str.contains('impl|constructor|student.java|fix|tostring|debug|studentrecordio|gui|field|manager', 
    case = False, na=False), 'Tag_type'] = 'Implementation'
  pre_process.loc[pre_process['Tag'].str.contains('test|coverage|black|box|BBTP|BB', 
    case = False, na=False), 'Tag_type'] = 'Testing'
  pre_process.loc[pre_process['Tag'].str.contains('style|pmd|styl|comments', 
    case = False, na=False), 'Tag_type'] = 'Styling'
  pre_process.loc[pre_process['Tag'].str.contains('design|diagram|readme', 
    case = False, na=False), 'Tag_type'] = 'Design'
  pre_process.loc[pre_process['Tag'].str.contains('Untagged', 
    case = False, na=False), 'Tag_type'] = 'Untagged'
  pre_process['Tag_type'].replace('', 'Misc', inplace = True)
  
  #clean message
  pre_process['Message_Type'] = ''
  pre_process.loc[pre_process['Message'].str.contains('document|javadoc', case = False, na=False),
    'Message_Type'] = 'Documentation'
  pre_process.loc[pre_process['Message'].str.contains('impl|constructor|student.java|fix|tostring|debug|studentrecordio|gui|field|manager', 
    case = False, na=False), 'Message_Type'] = pre_process['Message_Type'] + 'Implementation'
  pre_process.loc[pre_process['Message'].str.contains('test|coverage|black|box|BBTP|BB', 
    case = False, na=False), 'Message_Type'] = pre_process['Message_Type'] + 'Testing'
  pre_process.loc[pre_process['Message'].str.contains('style|pmd|styl|comments', 
    case = False, na=False), 'Message_Type'] = pre_process['Message_Type'] + 'Styling'
  pre_process.loc[pre_process['Message'].str.contains('design|diagram|readme', 
    case = False, na=False), 'Message_Type'] = pre_process['Message_Type'] + 'Design'
  pre_process['Message_Type'].replace('', 'Misc', inplace = True)
  return pre_process

#get accuracy of message/tag classifier
def compareTagMessageAccuracy(pre_process):
  pre_process['tagMessageAccuracy'] = [x[0] in x[1] for x in zip(pre_process['Tag_type'], 
    pre_process['Message_Type'])]
  pre_process.loc[pre_process.isTag == 'Untagged', 'tagMessageAccuracy'] = 'Untagged'
  totalTrue = pre_process['author_ID'][pre_process['tagMessageAccuracy'] == True].count()
  totalFalse = pre_process['author_ID'][pre_process['tagMessageAccuracy'] == False].count()
  accuracy = (totalTrue/(totalTrue + totalFalse)) * 100
  print('MessageTagClassifierAccuracy: ' + str(accuracy) + '%')
  return pre_process

#classify commit messages (Kunal's method)
def classify(pre_process):
  pre_process['Commit_Type'] = ''
  pre_process.loc[pre_process['Commit_message'].str.contains('document|javadoc', case = False, na=False),
    'Commit_Type'] = 'Documentation'
  pre_process.loc[pre_process['Commit_message'].str.contains('impl|constructor|student.java|fix|tostring|debug|studentrecordio|gui|field|manager', 
    case = False, na=False), 'Commit_Type'] = pre_process['Commit_Type'] + 'Implementation'
  pre_process.loc[pre_process['Commit_message'].str.contains('test|coverage|black|box|BBTP|BB', 
    case = False, na=False), 'Commit_Type'] = pre_process['Commit_Type'] + 'Testing'
  pre_process.loc[pre_process['Commit_message'].str.contains('style|pmd|styl|comments', 
    case = False, na=False), 'Commit_Type'] = pre_process['Commit_Type'] + 'Styling'
  pre_process.loc[pre_process['Commit_message'].str.contains('design|diagram|readme', 
    case = False, na=False), 'Commit_Type'] = pre_process['Commit_Type'] + 'Design'
  pre_process['Commit_Type'].replace('', 'Misc', inplace = True)
  return pre_process
  
#classify commit messages by updated version (Ruth's method)
def classifyRuth(pre_process):
  pre_process['Commit_Type_2'] = ''
  pre_process['Commit_tag_lower'] = pre_process.Tag.str.lower()
  pre_process.loc[pre_process['Commit_tag_lower'].str.contains('untag', na=False),
    'Commit_Type_2'] = 'untagged'
  pre_process.loc[pre_process['Commit_tag_lower'].str.contains('skeleton|skel|libraries|initial|collection|setup|start', na=False),
    'Commit_Type_2'] = 'skeleton'
  pre_process.loc[pre_process['Commit_tag_lower'].str.contains('fix', na=False),
    'Commit_Type_2'] = 'bugfix'
  pre_process.loc[pre_process['Commit_tag_lower'].str.contains('test|random|hope|jenk|junit|unit|bb|box', na=False),
    'Commit_Type_2'] = 'test'
  pre_process.loc[pre_process['Commit_tag_lower'].str.contains('debug|exception|throw|error|deb|dubug|degug|pmd', na=False),
    'Commit_Type_2'] = 'debugging'
  pre_process.loc[pre_process['Commit_tag_lower'].str.contains('impl|impel|mentation|menting|optimization|integra|fsm|linkedlist', na=False),
    'Commit_Type_2'] = 'implementation'
  pre_process.loc[pre_process['Commit_tag_lower'].str.contains('doc|deploy|comments', na=False),
    'Commit_Type_2'] = 'documentation'
  pre_process.loc[pre_process['Commit_tag_lower'].str.contains('style|design|gui|uml', na=False),
    'Commit_Type_2'] = 'style'
  pre_process.loc[pre_process['Commit_tag_lower'].str.contains('refactor|remove', na=False),
    'Commit_Type_2'] = 'cleanup'
  pre_process.loc[pre_process['Commit_tag_lower'].str.contains('git|revert', na=False),
    'Commit_Type_2'] = 'versioning'
  pre_process['Commit_Type_2'].replace('', 'misc', inplace = True)
  pre_process.drop('Commit_tag_lower', axis = 1, inplace = True)
  return pre_process

#get tag of commit message
#reference - https://stackoverflow.com/questions/46350705/creating-new-column-in-pandas-dataframe-using-regex/46351084
def getTagAndMessage(pre_process):
  search = []
  tag = []    
  message = []
  for value in pre_process['Commit_message']:
      result = (re.search(r"\[([A-Za-z0-9_]+)\]", str(value)))
      if (result != None):
        #spell check (only for documentation & implementation)
        '''word = result.group(1).upper()
        firstDI = False
        if (word[0] == 'D' or word[0] == 'I'):
          firstDI = True
        if (firstDI == True):
          if (dict.check(word) == False):
            list = dict.suggest(word)
            if len(list) > 0:
              if list[0].upper() == 'DOCUMENTATION':
                search.append('DOCUMENTATION')
              elif list[0].upper() == 'IMPLEMENTATION':
                search.append('IMPLEMENTATION')
              else:
                search.append(word)
            else:
              search.append('Unknown')
          else:
            search.append(word)
        else:    
          search.append(word)'''
        message.append(re.sub(r"\[([A-Za-z0-9_]+)\]", '', str(value)))
        search.append(result.group(1).upper())
        tag.append('Tagged')
      else:
        message.append(str(value))
        search.append('Untagged')
        tag.append('Untagged')
  pre_process['Tag'] = search
  pre_process['Message'] = message
  pre_process['isTag'] = tag
  return pre_process

#drop rows with missing gender
def dropGender(pre_process):
  pre_process = pre_process.dropna()
  pre_process = pre_process[pre_process.Gender != '0']
  return pre_process

#merge two csv files and preprocess the dataframe
def merge_process(gender, commits):
  gender = pd.read_csv(gender)
  gender = gender.rename(columns={'Unique_ID': 'author_ID'})
  gender = gender.fillna('U.S')
  commits = pd.read_csv(commits)
  pre_process = pd.merge(commits, gender, on = 'author_ID', how = 'inner')
  pre_process = pre_process.drop(pre_process.columns[12], axis = 1)
  pre_process = pre_process.rename(columns = {' Commit_datetime': 'Commit_datetime'})
  pre_process = pre_process[['author_ID', 'Gender', 'Repo_name', 'Commit_datetime', 
    'Commit_message', 'Added', 'Deleted', 'Modified', 'Removed']]
  pre_process['Commit_datetime'] = pre_process['Commit_datetime'].str[:-6]
  pre_process['Commit_datetime'] = pd.to_datetime(pre_process['Commit_datetime'], 
    format = '%Y-%m-%d %H:%M:%S', errors = 'coerce')
  pre_process['Day'] = pre_process['Commit_datetime'].dt.dayofweek
  pre_process['MSG_LEN'] = pre_process['Commit_message'].str.len()
  pre_process = pre_process.sort_values('Repo_name')
  return pre_process

def teamGenderInfo(pre_process):
  #Add columns/features like mixed team: all_male, all_female, mixed, contains_U.S,  how many females in the team 0, 1, 2, 3…, 
  repo_gender = pre_process.groupby(['Repo_name'])['Gender'].agg(['unique']).reset_index()
  unique_gender = pre_process.groupby(['Repo_name', 'author_ID'])['Gender'].first().reset_index()
  gender_count = unique_gender.groupby('Repo_name')['Gender'].apply(list).reset_index()
  gender_count.Gender = (gender_count.Gender).astype(str).str[1:-1]
  gender_count['count_of_females'] = gender_count.Gender.str.count('F')
  gender_count['count_of_males'] = gender_count.Gender.str.count('M')
  gender_count['count_of_US'] = gender_count.Gender.str.count('U.S')
  gender_count['team_info'] = 'mixed'
  gender_count.loc[~(gender_count['Gender'].str.contains('M', na=False)) & ~(gender_count['Gender'].str.contains('U.S', na=False)) ,
    'team_info'] = 'all_female'
  gender_count.loc[~(gender_count['Gender'].str.contains('F', na=False)) & ~(gender_count['Gender'].str.contains('U.S', na=False)) ,
    'team_info'] = 'all_male'
  gender_count.loc[gender_count['Gender'].str.contains('U.S', na=False),
    'team_info'] = 'contains_U.S'
  gender_pair_dict = dict(zip(gender_count.Repo_name, gender_count.team_info))
  gender_countF_dict = dict(zip(gender_count.Repo_name, gender_count.count_of_females))   
  gender_countM_dict = dict(zip(gender_count.Repo_name, gender_count.count_of_males)) 
  pre_process["team_info"] = pre_process.Repo_name.map(gender_pair_dict)
  pre_process["count_of_females"]  = pre_process.Repo_name.map(gender_countF_dict)
  pre_process["count_of_males"] = pre_process.Repo_name.map(gender_countM_dict)
  return pre_process

def NLPModel(pre_process):
  #####   NLP BEGINS   ############## 
  #https://www.analyticsvidhya.com/blog/2019/04/predicting-movie-genres-nlp-multi-label-classification/
  # In pre_process df we are using Commit_message to predict the Commit_Type_Grouped (untagged and misc)
  pre_process['Commit_messageWTags'] = pre_process['Commit_message'].str.replace('[', '')
  pre_process['Commit_messageWTags'] = pre_process['Commit_messageWTags'].str.replace(']', '')
  # function for text cleaning 
  def clean_text(text):
      # remove backslash-apostrophe 
      text = re.sub("\'", "", text) 
      # remove everything except alphabets 
      text = re.sub("[^a-zA-Z]"," ",text) 
      # remove whitespaces 
      text = ' '.join(text.split()) 
      # convert text to lowercase 
      text = text.lower()     
      return text    
  #Train with original commit messages
  pre_process['clean_cmWOTags'] = pre_process['Commit_messageWTags'].apply(lambda x: clean_text(str(x)))
  #Frequency of words in the commit message column
  def freq_words(x, terms = 30): 
      all_words = ' '.join([text for text in x]) 
      all_words = all_words.split() 
      fdist = nltk.FreqDist(all_words) 
      words_df = pd.DataFrame({'word':list(fdist.keys()), 'count':list(fdist.values())}) 
      # selecting top 20 most frequent words 
      d = words_df.nlargest(columns="count", n = terms) 
      # visualize words and frequencies
      '''plt.figure(figsize=(12,15)) 
      ax = sns.barplot(data=d, x= "count", y = "word") 
      ax.set(ylabel = 'Word') 
      plt.show()'''     
  # remove stopwords 
  from nltk.corpus import stopwords
  stop_words = set(stopwords.words('english'))
  # function to remove stopwords
  def remove_stopwords(text):
      no_stopword_text = [w for w in text.split() if not w in stop_words]
      return ' '.join(no_stopword_text)
  pre_process['clean_cmWOTags'] = pre_process['clean_cmWOTags'].apply(lambda x: remove_stopwords(x))
  freq_words(pre_process['clean_cmWOTags'], 100)   
  #Remove test dataset - untagged and misc
  train_val_set = pre_process[(pre_process.Commit_Type_2 != 'untagged') & (pre_process.Commit_Type_2!='misc')]
  train_val_set.Commit_Type_2.unique()
  train_val_set.Commit_Type_2 = train_val_set.Commit_Type_2.str.replace('_',', ')
  train_val_set.Commit_Type_2 = train_val_set.Commit_Type_2.str.lstrip(', ')
  train_val_set.Commit_Type_2 = "["+train_val_set.Commit_Type_2+"]" 
  # Using TF-IDF features for feature extraction max_df means ignore words 20% less popular
  tfidf_vectorizer = TfidfVectorizer(max_df = 0.7, max_features = 1500)
  # split dataset into training and validation set
  xtrain, xval, ytrain, yval = train_test_split(train_val_set['clean_cmWOTags'], train_val_set['Commit_Type_2'], test_size=0.5, random_state=9)
  # create TF-IDF features
  xtrain_tfidf = tfidf_vectorizer.fit_transform(xtrain)
  xval_tfidf = tfidf_vectorizer.transform(xval)
  #fit model
  lr = LogisticRegression()
  clf = OneVsRestClassifier(lr)
  # fit model on train data
  clf.fit(xtrain_tfidf, ytrain)
  # make predictions for validation set
  y_pred = clf.predict(xval_tfidf)
  #eval performance
  '''f1_score(yval, y_pred, average="micro")'''
  #clean the rest commit meesages and predict the all commit messages
  all_tfidf = tfidf_vectorizer.transform(pre_process['Commit_messageWTags'].apply(lambda x: clean_text(str(x))))
  all_pred = clf.predict(all_tfidf)
  #Final dataframe
  pre_process['predicted_tags'] = all_pred
  #drop irrelevant columns
  pre_process.drop(['Commit_messageWTags', 'clean_cmWOTags'], axis = 1, inplace = True)
  return pre_process

def mixed(pre_process):  
  femaleTeams = 0
  maleTeams = 0
  mixedTeams = 0
  femaleCommits = 0
  femaleLinesAdded = 0
  femaleLinesModified = 0
  maleCommits = 0
  maleLinesAdded = 0
  maleLinesModified = 0
  mixedTeams = 0
  mixedCommits = 0
  mixedLinesAdded = 0
  mixedFCommits = 0
  mixedFLinesAdded = 0
  mixedFLinesModified = 0
  mixedMCommits = 0
  mixedMLinesAdded = 0
  mixedMLinesModified = 0
  repo = 'csc216-T-P2-0'
  mixedTeamsDF = pre_process
  mixedTeamsDF = mixedTeamsDF.iloc[0:0]
  for i in range(1,10):
    partners = pre_process['Gender'][pre_process['Repo_name'] == repo + str(i)].unique()
    if (len(partners) == 2):
      if (partners[0] != 'U.S' and partners[1] != 'U.S'):
        mixedTeamsDF = mixedTeamsDF.append(pre_process[pre_process['Repo_name'] == repo + str(i)])
        mixedTeams += 1
        mixedCommits += pre_process[pre_process['Repo_name'] == repo + str(i)].shape[0]
        mixedMCommits += pre_process[pre_process['Gender'] == 'M'][pre_process['Repo_name'] 
          == repo + str(i)].shape[0]
        mixedFCommits += pre_process[pre_process['Gender'] == 'F'][pre_process['Repo_name'] 
          == repo + str(i)].shape[0]
        mixedMLinesAdded += pre_process['Added'][pre_process['Gender'] == 'M'][pre_process['Repo_name'] 
          == repo + str(i)].sum()
        mixedMLinesModified += pre_process['Modified'][pre_process['Gender'] == 'M'][pre_process['Repo_name'] 
          == repo + str(i)].sum()
        mixedFLinesAdded += pre_process['Added'][pre_process['Gender'] == 'F'][pre_process['Repo_name'] 
          == repo + str(i)].sum()
        mixedFLinesModified += pre_process['Modified'][pre_process['Gender'] == 'F'][pre_process['Repo_name'] 
          == repo + str(i)].sum()
    elif partners[0] == 'M':
      maleTeams += 1
      maleCommits += pre_process[pre_process['Repo_name'] == repo + str(i)].shape[0]
      maleLinesAdded += pre_process['Added'][pre_process['Repo_name'] == repo + str(i)].sum()
      maleLinesModified += pre_process['Modified'][pre_process['Repo_name'] == repo + str(i)].sum()
    elif partners[0] == 'F':
      femaleTeams += 1
      femaleCommits += pre_process[pre_process['Repo_name'] == repo + str(i)].shape[0]
      femaleLinesAdded += pre_process['Added'][pre_process['Repo_name'] == repo + str(i)].sum()
      femaleLinesModified += pre_process['Modified'][pre_process['Repo_name'] == repo + str(i)].sum()    
  repo = 'csc216-T-P2-'
  for i in range(10,72):
    partners = pre_process['Gender'][pre_process['Repo_name'] == repo + str(i)].unique()
    if (len(partners) == 2):
      if (partners[0] != 'U.S' and partners[1] != 'U.S'):
        mixedTeamsDF = mixedTeamsDF.append(pre_process[pre_process['Repo_name'] == repo + str(i)])
        mixedTeams+= 1
        mixedCommits += pre_process[pre_process['Repo_name'] == repo + str(i)].shape[0]
        mixedMCommits += pre_process[pre_process['Gender'] == 'M'][pre_process['Repo_name'] 
          == repo + str(i)].shape[0]
        mixedFCommits += pre_process[pre_process['Gender'] == 'F'][pre_process['Repo_name'] 
          == repo + str(i)].shape[0]
        mixedMLinesAdded += pre_process['Added'][pre_process['Gender'] == 'M'][pre_process['Repo_name'] 
          == repo + str(i)].sum()
        mixedFLinesAdded += pre_process['Added'][pre_process['Gender'] == 'F'][pre_process['Repo_name'] 
          == repo + str(i)].sum()
        mixedMLinesModified += pre_process['Modified'][pre_process['Gender'] == 'M'][pre_process['Repo_name'] 
          == repo + str(i)].sum()
        mixedFLinesModified += pre_process['Modified'][pre_process['Gender'] == 'F'][pre_process['Repo_name'] 
          == repo + str(i)].sum()
    elif partners[0] == 'M':
      maleTeams+= 1
      maleCommits += pre_process[pre_process['Repo_name'] == repo + str(i)].shape[0]
      maleLinesAdded += pre_process['Added'][pre_process['Repo_name'] == repo + str(i)].sum()
      maleLinesModified += pre_process['Modified'][pre_process['Repo_name'] == repo + str(i)].sum()
    elif partners[0] == 'F':
      femaleTeams+= 1
      femaleCommits += pre_process[pre_process['Repo_name'] == repo + str(i)].shape[0]
      femaleLinesAdded += pre_process['Added'][pre_process['Repo_name'] == repo + str(i)].sum()
      femaleLinesModified += pre_process['Modified'][pre_process['Repo_name'] == repo + str(i)].sum()
  return mixedTeamsDF