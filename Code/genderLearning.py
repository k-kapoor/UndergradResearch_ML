'''
  File name: preprocess.py
  Author: Kunal Kapoor
  Date Created: 3/11/2021
  Date last modified: 4/5/2021
'''


#IMPORTS
from preprocessing import merge_process, dropGender, splitByRepo, getTagAndMessage, classify
from preprocessing import classifyTagMessage, classifyRuth, compareTagMessageAccuracy
from preprocessing import teamGenderInfo, NLPModel, mixed
from models import DTClassifyGender, DTClassifyCommit, LogRModel, NBModel, KNNModel, SVMModel
import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from sklearn import preprocessing


#PREPROCESSING
# merge gender & commits csv file and preprocess dataframe
pre_process = merge_process('../csv/Gender.csv', '../csv/commits.csv')
pre_process.fillna('U.S')
#Get tag of commit messages, and column for just messages
getTagAndMessage(pre_process)
#classify tags and methods seperately, then compare accuracy
pre_process = classifyTagMessage(pre_process)
pre_process = compareTagMessageAccuracy(pre_process)
#classify message (based on decided keywords)
pre_process = classify(pre_process)
pre_process = classifyRuth(pre_process)
#classify commits by repo
pre_process = splitByRepo(pre_process)
#get gender info for teams (all male/female, etc.)
pre_process = teamGenderInfo(pre_process)
#Use NLP to classify commit messages
pre_process = NLPModel(pre_process)
#get dataframe of team p2 commits
mixedTeamsDF = mixed(pre_process)


#MACHINE LEARNING
#numerize categorical columns for the DT
labelEnc = preprocessing.LabelEncoder()
mixedTeamsDF['Enc_Tag'] = labelEnc.fit_transform(mixedTeamsDF['isTag'])
mixedTeamsDF['Enc_Commit_Type'] = labelEnc.fit_transform(mixedTeamsDF['Commit_Type'])
mixedTeamsDF['BinarizedGender'] = labelEnc.fit_transform(mixedTeamsDF['Gender'])
#call ML Models
'''DTClassifyGender(mixedTeamsDF)
DTClassifyCommit(mixedTeamsDF)
LogRModel(mixedTeamsDF)
NBModel(mixedTeamsDF)
KNNModel(mixedTeamsDF)
SVMModel(mixedTeamsDF)'''



#SAVE CSV
mixedTeamsDF.drop(['Enc_Tag', 'Enc_Commit_Type', 'BinarizedGender'], axis = 1, inplace = True)
mixedTeamsDF.to_csv('../csv/mixedTeams.csv')
pre_process.to_csv('../csv/pre_processed.csv')