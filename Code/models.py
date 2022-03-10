'''
  File name: mlmodels.py
  Author: Kunal Kapoor
  Date Created: 4/6/2021
  Date last modified: 4/6/2021
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import CategoricalNB
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn import svm
import numpy as np
from sklearn import tree
from sklearn import metrics
import pydotplus
from IPython.display import Image
import graphviz
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

#Export DT tree as png file
def exportTree(DTtree, features, classes):
  dot_data = tree.export_graphviz(DTtree, out_file = None, filled = True, proportion = True,
    feature_names = features, class_names = classes, max_depth = 3)
  import os
  os.environ["PATH"] += os.pathsep + 'C:\Program Files\Graphviz2\bin'
  graph = pydotplus.graph_from_dot_data(dot_data)
  graph.write_png('../images/tree.png')

#decision tree classify gender
def DTClassifyGender(mixedTeamsDF): 
  x = mixedTeamsDF[['Added', 'Deleted', 'Modified', 'Removed', 'Enc_Commit_Type', 'Enc_Tag', 'Day']]
  features = ['Added', 'Deleted', 'Modified', 'Removed', 'Enc_Commit_Type', 'Enc_Tag', 'Day']
  classes = ['M', 'F']
  y = mixedTeamsDF[['Gender']]
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
  x_train1, x_validation, y_train1, y_validation = train_test_split(x_train, y_train,
    test_size = .2, random_state = 1)
  decisionTree = tree.DecisionTreeClassifier(class_weight = 'balanced', 
    criterion = 'entropy', random_state = 1)
  genderClf = decisionTree.fit(x_train1, y_train1)
  y_pred = genderClf.predict(x_validation)
  y_testpred = genderClf.predict(x_test)
  print('\nDepth of Decision Tree: ',genderClf.get_depth())
  print("Classify Gender DT Balanced Accuracy: ", metrics.balanced_accuracy_score(y_test, y_testpred))
  print("Classify Gender DT F-Score: ", metrics.f1_score(y_test, y_testpred, pos_label = 'M'))
  print('Classify gender DT most important features: ', genderClf.feature_importances_, '\n')
  exportTree(genderClf, features, classes)
  
#decision tree classify commit type
def DTClassifyCommit(mixedTeamsDF):
  x = mixedTeamsDF[['Added', 'Deleted', 'Modified', 'Removed', 'BinarizedGender', 'Enc_Tag', 'Day']]
  y = mixedTeamsDF[['Enc_Commit_Type']]
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
  x_train1, x_validation, y_train1, y_validation = train_test_split(x_train, y_train,
    test_size = .2, random_state = 1)
  decisionTree = tree.DecisionTreeClassifier(criterion = 'entropy', random_state = 1)
  genderClf = decisionTree.fit(x_train1, y_train1)
  y_pred = genderClf.predict(x_validation)
  y_testpred = genderClf.predict(x_test)
  print("Decision Tree classify commit type")
  print("Balanced accuracy:", metrics.balanced_accuracy_score(y_test, y_testpred))
  print("Classify Commit DT F-Score: ", metrics.f1_score(y_test, y_testpred, average = 'micro'))
  print('Classify Commit Type DT most important features: ', genderClf.feature_importances_, '\n')

#use logistic regression to classify gender
def LogRModel(mixedTeamsDF):
  normalized_df = mixedTeamsDF[['Added', 'Deleted', 'Modified', 'Removed', 
    'Enc_Commit_Type', 'Enc_Tag', 'Day']]
  x = (normalized_df - normalized_df.min()) / (normalized_df.max() - normalized_df.min())
  y = mixedTeamsDF[['Gender']]
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
  x_train1, x_validation, y_train1, y_validation = train_test_split(x_train, y_train,
    test_size = .2, random_state = 1)
  regression = LogisticRegression(class_weight = 'balanced', C = .3, random_state = 1)
  regression = regression.fit(x_train1 / np.std(x_train1, 0), y_train1)
  y_testpred = regression.predict(x_test)
  print("Logistic Regression Balanced Accuracy: ", metrics.balanced_accuracy_score(y_test, y_testpred))
  print("Logistic Regression F Score: ", metrics.f1_score(y_test, y_testpred, pos_label = 'M'))
  print('LogR coefficients', regression.coef_, '\n')
 
#use various naive bayes classifiers to classify gender 
def NBModel(mixedTeamsDF):
  x = mixedTeamsDF[['Added', 'Deleted', 'Modified', 'Removed', 'Enc_Commit_Type', 'Enc_Tag', 'Day']]
  y = mixedTeamsDF[['Gender']]
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
  x_train1, x_validation, y_train1, y_validation = train_test_split(x_train, y_train,
    test_size = .2, random_state = 1)
  naive = GaussianNB().fit(x_train1, y_train1)
  mnb = MultinomialNB().fit(x_train1, y_train1)
  bern = BernoulliNB().fit(x_train1, y_train1)
  CNB = CategoricalNB().fit(x_train1, y_train1)
  cmp = ComplementNB().fit(x_train1, y_train1)
  y_testpred = naive.predict(x_test)
  print('Naive Bayes: ')
  print("GNB Balanced Accuracy: ", metrics.balanced_accuracy_score(y_test, y_testpred))
  print("GNB F Score: ", metrics.f1_score(y_test, y_testpred, pos_label = 'M'))
  y_testpred = mnb.predict(x_test)
  print("MNB Balanced Accuracy: ", metrics.balanced_accuracy_score(y_test, y_testpred))
  print("MNB F Score: ", metrics.f1_score(y_test, y_testpred, pos_label = 'M'))
  y_testpred = bern.predict(x_test)
  print("BNB Balanced Accuracy: ", metrics.balanced_accuracy_score(y_test, y_testpred))
  print("BNB F Score: ", metrics.f1_score(y_test, y_testpred, pos_label = 'M'))
  y_testpred = CNB.predict(x_test)
  print("CNB Balanced Accuracy: ", metrics.balanced_accuracy_score(y_test, y_testpred))
  print("CNB F Score: ", metrics.f1_score(y_test, y_testpred, pos_label = 'M'))
  y_testpred = cmp.predict(x_test)
  print("CMP Balanced Accuracy: ", metrics.balanced_accuracy_score(y_test, y_testpred))
  print("CMP F Score: ", metrics.f1_score(y_test, y_testpred, pos_label = 'M'), '\n')

#KNN/Radius Neighbors (Classify Gender)
def KNNModel(mixedTeamsDF):
  x = mixedTeamsDF[['Added', 'Deleted', 'Modified', 'Removed', 'Enc_Commit_Type', 'Enc_Tag', 'Day']]
  y = mixedTeamsDF[['Gender']]
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
  x_train1, x_validation, y_train1, y_validation = train_test_split(x_train, y_train,
    test_size = .2, random_state = 1)
    # code used from - https://data-flair.training/blogs/machine-learning-algorithms-in-python/
  k_range = range(1, 30) 
  scoresK = []
  scoresR = []
  fscoreK= []
  fscoreR = []
  for k in k_range:
    y_predK = KNeighborsClassifier(n_neighbors = k, weights = 'distance', p = 3).fit(x_train1, 
      y_train1).predict(x_test)
    y_predR = RadiusNeighborsClassifier(radius = k, weights = 'distance', p = 3,
      outlier_label = 'most_frequent').fit(x_train1, y_train1).predict(x_test)
    scoresK.append(metrics.balanced_accuracy_score(y_test, y_predK))
    scoresR.append(metrics.balanced_accuracy_score(y_test, y_predR))
    fscoreK.append(metrics.f1_score(y_test, y_predK, pos_label = 'M'))
    fscoreR.append(metrics.f1_score(y_test, y_predR, pos_label = 'M'))
  print('KNN Balanced Accuracy: ' + str(max(scoresK)))
  print('Radius Balanced Accuracy: ' + str(max(scoresR)) + '\n')
  print('KNN F Score: ' + str(max(fscoreK)))
  print('Radius F Score: ' + str(max(fscoreR)) + '\n')
  plt.plot(k_range,scoresK)
  plt.xlabel('k for kNN')
  plt.ylabel('Testing Accuracy')
  plt.show()
  
#use SVM model to classify gender
def SVMModel(mixedTeamsDF):
  x = mixedTeamsDF[['Added', 'Deleted', 'Modified', 'Removed', 'Enc_Commit_Type', 'Enc_Tag', 'Day']]
  y = mixedTeamsDF[['Gender']]
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 1)
  x_train1, x_validation, y_train1, y_validation = train_test_split(x_train, y_train,
    test_size = .2, random_state = 1)
  y_pred = svm.LinearSVC(class_weight = 'balanced', C = .005, random_state = 1).fit(x_train, 
    y_train).predict(x_test)
  print("SVM Balanced Accuracy: ", metrics.balanced_accuracy_score(y_test, y_pred))
  print("SVM F Score: ", metrics.f1_score(y_test, y_pred, pos_label = 'M'))