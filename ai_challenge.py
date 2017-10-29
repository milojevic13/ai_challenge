# -*- coding: utf-8 -*-
"""
@author: Milica Milojevic
"""
#==============================================================================
# Steps to perform the task:
#     1.Data gathering: importing the data
#     2.Data pre-processing: perform data cleaning (e.g. incomplete data),attribute selection, etc
#     3.Data mining: split the data into training and test data, apply data mining techniques on the test data
#     4.Perform evaluation of the data mining techniques (e.g. accuracy of the data technique)
#==============================================================================


#==============================================================================
# Data gathering
#==============================================================================
#Import packages
import numpy as np
import pandas as pd
#Import file
data_terr_attack = pd.read_csv("gtd_1970_1994.csv",encoding='ISO-8859-1')

# Remove rows of the dataframe for which gname is not defined
data_terr_attack_90 = data_terr_attack[data_terr_attack.gname != "Unknown"]

#NOTE: Reduce the amount of data stored in a dataframe since there is no enough RAM memory to process all data on the used laptop  
data_terr_attack_90 = data_terr_attack_90[data_terr_attack_90.iyear >= 1990]

# Limit long strings. 
def shortening(df):
    
    df.loc[df['weaptype1_txt'] == 'Vehicle (not to include vehicle-borne explosives, i.e., car or truck bombs)', 'weaptype1_txt'] = 'Vehicle'
    df.loc[df['propextent_txt'] == 'Minor (likely < $1 million)','propextent_txt'] = 'Minor (< $1 million)'
    df.loc[df['propextent_txt'] == 'Major (likely > $1 million but < $1 billion)','propextent_txt'] = 'Major (< $1 billion)'
    df.loc[df['propextent_txt'] == 'Catastrophic (likely > $1 billion)', 'propextent_txt'] = 'Catastrophic (> $1 billion)'

    return df
data_terr_attack_90 = shortening(data_terr_attack_90)

#-9 & -99 are unknowns replaced with nan
def nines(df):
    for i in ["ishostkid", "property","INT_LOG","INT_IDEO","INT_MISC","INT_ANY","claimed"]:
         df.loc[df[i] == -9,i]=df.loc[df[i] == -9,i].replace(-9,np.nan)
    return df  
data_terr_attack_90 = nines(data_terr_attack_90)

def ninty_nines(df):
    for i in ["nhostkid","nhostkidus","nhours","nperpcap","nperps"]: 
        df.loc[df[i] == -99,i]=df.loc[df[i] == -99,i].replace(-99,np.nan)
    return df  
data_terr_attack_90 = ninty_nines(data_terr_attack_90)



#==============================================================================
# Data pre-processing
#==============================================================================

#Checking the number of nan values per column, in order to select appropriate atributes
feature_null_counts = data_terr_attack_90.apply(lambda col: col.isnull().sum(), axis = 0)

#Since dataframe is composed of text attributes and numeric attributes, the pre-processing step for these 
#attributes is diferent. Thus, attributes are first separated into text attributes and the 
#numeric attributes by observing the values of 'feature_null_counts':


#We select both numeric and text attributes only in case there is significant number of non nan values. Additionally some of the attributes like for example
#'targettyp1' and 'targettype1_txt' are saying the same thing in different form, thus we choose only one of them 
att_num = ['iyear','imonth','iday','extended','country','region','specificity',
         'vicinity','crit1','crit2','crit3','doubtterr','alternative','multiple',
         'success','suicide','attacktype1','targtype1','targsubtype1','natlty1',
         'guncertain1','individual','nperps','weaptype1','weapsubtype1','nkill',
         'nkillus','nkillter','nwound','nwoundus','nwoundte','property','propextent',
          'propvalue','ishostkid','ransom','INT_LOG','INT_IDEO','INT_MISC','INT_ANY']


att_text = ['provstate','city','location','corp1','target1','dbsource','weapdetail']

#Define a class 'gname'
class_gname = data_terr_attack_90.gname

data_terr_attack_90 = data_terr_attack_90[att_num+att_text]

#Import packages for data pre-processing
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import FunctionTransformer
from sklearn.cross_validation import train_test_split

#Create an empty list in which we will store different pipelines
pipelines = []

#Use 'train_test_split' to split the data into train and test data.  
X_train, X_test, y_train, y_test = train_test_split(data_terr_attack_90, class_gname, test_size = 0.33, random_state = 1)

#Define a function for text attributes that fills all nan values with '' 
def process_att_text(col):
    
    def nest_function(df):

        return df[col].fillna("")
    
    return nest_function

#Define a function that processes the numeric attributes
def process_att_num(df):
    df = df[att_num]
    
    #Select columns with numeric attributes
    numeric_col = ['iyear','imonth','iday','nperps','nkill','nkillus','nkillter','nwound','nwoundus','nwoundte','propvalue']
   
    #Replace nulls with medains
    for i in numeric_col:
        df.ix[df[i].isnull(), i] = df[i].median()
   
    #Seclet binary columns
    binary_col = ['extended','country','region','specificity','vicinity','crit1',
                   'crit2','crit3','doubtterr','alternative','multiple','success',
                   'suicide','attacktype1','targtype1','targsubtype1','natlty1',
                   'guncertain1','individual','weaptype1','weapsubtype1','property',
                   'propextent','ishostkid','ransom','INT_LOG','INT_IDEO','INT_MISC','INT_ANY']
     
    #Replace null values with a radnom sample
    for i in binary_col:
        df.loc[df[i].isnull(), i] = np.random.choice([0,1],size = df[i].isnull().sum())       
    return df    


# Build a Document Term Matrix using CountVectorizer for all text attributes defined in att_text list.
for i in att_text:
    #FunctionTransformer turns any function into a transformer
    pipeline = make_pipeline(FunctionTransformer(process_att_text(i),validate = False),CountVectorizer(decode_error = 'ignore'))
    #Add pipeline to pipelines list
    pipelines.append(pipeline)

#Add the numeric data into the pipeline
pipelines.append(FunctionTransformer(process_att_num, validate = False))

#Make union of the pipelines
union = make_union(*pipelines)

#Fit and transform the training data
X_train_new = union.fit_transform(X_train)
#Transform test data
X_test_new = union.transform(X_test)


#==============================================================================
# Data mining
#==============================================================================
#Import packages
from sklearn.grid_search import GridSearchCV #for tuning the hyper-parameters of the estimator
from sklearn.metrics import accuracy_score #for evaluating the accuracy of the classifier
from sklearn.model_selection import cross_val_score #for evaluating a score by cross-validation
#Create an empty dictionary for storing the accuracy scores from different data mining techniques
accuracy_dict={}

#==============================================================================
# 1.K-nearest neighbour (KNN) classifier 
#http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
#==============================================================================
#Import packages 
from sklearn.neighbors import KNeighborsClassifier
#Check parameters of the classifier and their default values
KNeighborsClassifier().get_params
#Perform tuning hyper-parameters of the etsimator using GridSearchCV
n_list = list(range(3, 30))
#Perform GridSearchCV for classifier
knn_gscv= GridSearchCV(KNeighborsClassifier(), param_grid = {'n_neighbors':n_list})
#Perform fit method
knn_gscv.fit(X_train_new, y_train)
#Check the best parameter
print (knn_gscv.best_params_)

#Create an instance of the model
knn_model = KNeighborsClassifier(n_neighbors=3) 
#Fit the model using training data X_train_new and y_train as target values
knn_model.fit(X_train_new,y_train)
#Predict the class labels for the test data X_test_new
knn_predict = knn_model.predict(X_test_new)
#Check accuracy classifications score
#http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html
#and store it in the accurcay dictionary accuracy_dict
accuracy_dict['knn'] = accuracy_score(y_test, knn_predict)
#Pefrom cross-validation
knn_score = cross_val_score(knn_model, X_test_new, y_test, cv = 10)

#==============================================================================
# 2. Naive (Gaussian) Bayesian classifier 
#http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
#==============================================================================
#Import packages 
from sklearn.naive_bayes import GaussianNB
#Create an instance of the model
naive_bay_model = GaussianNB()
#Fit the model using training data X_train_new and y_train as target values
naive_bay_model.fit(X_train_new.toarray(),y_train) #In order to avoid getting the "TypeError: A sparse matrix was passed, but dense data is required. Use X.toarray() to convert to a dense numpy array." we use X_train_dtm and we write X_train_dtm.toarray()
#Predict the class labels for the test data X_test_new
naive_bay_predict = naive_bay_model.predict(X_test_new.toarray())
#Check accuracy classifications and store it in the accurcay dictionary accuracy_dict
accuracy_dict['naive_bay'] = accuracy_score(y_test, naive_bay_predict)
#Pefrom cross-validation
naive_bay_score = cross_val_score(naive_bay_model, X_test_new.toarray(), y_test, cv = 10)


#==============================================================================
# 3.Decision Tree Classifier 
#http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
#====================================================== ========================
#Import packages
from sklearn.tree import DecisionTreeClassifier
#Check parameters of the classifier
DecisionTreeClassifier().get_params
#Perform GridSearchCV for classifier
dec_tr_gscv = GridSearchCV(DecisionTreeClassifier(),param_grid = {'criterion': ['gini','entropy']})
#Fit the model using training data X_train_new and y_train as target values
dec_tr_gscv.fit(X_train_new, y_train)
#Check the best parameter
print (dec_tr_gscv.best_params_)

#Create an instance of the model
dec_tr_model = DecisionTreeClassifier(criterion='entropy') 
#Fit the model using training data X_train_new and y_train as target values
dec_tr_model.fit(X_train_new,y_train)
#Predict the class labels for the test data X_test_new
dec_tr_predict = dec_tr_model.predict(X_test_new)
#Check accuracy classifications and store it in the accurcay dictionary accuracy_dict
accuracy_dict['dec_tr'] = accuracy_score(y_test, dec_tr_predict)
#Pefrom cross-validation
dec_tr_score = cross_val_score(dec_tr_model, X_test_new, y_test, cv = 10)

#==============================================================================
# 4.Random Forest Classifier 
#http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
#==============================================================================
#Import packages
from sklearn.ensemble import RandomForestClassifier
#Check parameters of the classifier
RandomForestClassifier().get_params
#Define a range of parameter 'n_estimators'
n_estim = [10,50,100]
#Perform GridSearchCV for classifier
rand_for_gscv = GridSearchCV(RandomForestClassifier(),param_grid = {'criterion': ['gini','entropy'], 'n_estimators':n_estim})
#Fit the model using training data X_train_new and y_train as target values
rand_for_gscv.fit(X_train_new,y_train)
print (rand_for_gscv.best_params_)

#Create an instance of the model
rand_for_model = RandomForestClassifier(n_estimators=100)
#Fit the model using training data X_train_new and y_train as target values
rand_for_model.fit(X_train_new,y_train)
#Predict the class labels for the test data X_test_dim
rand_for_predict = rand_for_model.predict(X_test_new)
#Check accuracy classifications and store it in the accurcay dictionary accuracy_dict
accuracy_dict['rand_for'] = accuracy_score(y_test, rand_for_predict)
#Pefrom cross-validation
rand_for_score = cross_val_score(rand_for_model, X_test_new, y_test, cv = 10)  #cv- k-fold

#==============================================================================
# 5.Stochastic Gradient Descent (SGD) classifier 
#http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html
#==============================================================================
#Import packages
from sklearn.linear_model import SGDClassifier
#Create an instance of the model
sgd_model = SGDClassifier()
#Fit the model using training data X_train_new and y_train as target values
sgd_model.fit(X_train_new,y_train)
#Predict the class labels for the test data X_test_new
sgd_predict = rand_for_model.predict(X_test_new)
#Check accuracy classifications and store it in the accurcay dictionary accuracy_dict
accuracy_dict['sgd'] = accuracy_score(y_test, sgd_predict)
#Pefrom cross-validation
sgd_score = cross_val_score(sgd_model, X_test_new, y_test, cv = 10)  #cv- k-fold

#==============================================================================
# 6.Linear Support Vector Classification
#==============================================================================
#Import packages
from sklearn.svm import LinearSVC
#Create an instance of the model
lin_svc_model = LinearSVC()
#Fit the model using training data X_train_new and y_train as target values
lin_svc_model.fit(X_train_new,y_train)
#Predict the class labels for the test data X_test_new
lin_svc_predict = lin_svc_model.predict(X_test_new)
#Check accuracy classifications and store it in the accurcay dictionary accuracy_dict
accuracy_dict['lin_svc'] = accuracy_score(y_test, lin_svc_predict)
#Pefrom cross-validation
lin_svc_score = cross_val_score(lin_svc_model, X_test_new, y_test, cv = 10) 

#==============================================================================
# 7.Logistic regression Classifier    
#==============================================================================
#Import packages
from sklearn.linear_model import LogisticRegression

#Create an instance of the model
log_reg_model = LogisticRegression() 
#Fit the model using training data X_train_new and y_train as target values
log_reg_model.fit(X_train_new,y_train)
#Predict the class labels for the test data X_test_new
log_reg_predict = log_reg_model.predict(X_test_new)
#Check accuracy classifications and store it in the accurcay dictionary accuracy_dict
accuracy_dict['lin_svc']= accuracy_score(y_test, log_reg_predict) 
#Pefrom cross-validation
log_reg_score = cross_val_score(log_reg_model, X_test_new, y_test, cv = 10) 


#==============================================================================
# Evaluate classifiers by observing their accuracy (accurcay_dict) and cross-validation (score_table)
#==============================================================================
#Create dataframe for evaluating a score by cross-validation 
score_table = pd.DataFrame(columns = ['model', 'cv_10'])
#Define a list of evaluated models
models = ['KNN Classifier', 'Naive Bayesian', 'Decision Tree', 'Random Forest Classifier', 'SGD Classifier','Linear SVM','Logistic Regression']
#Insert a mean value of each model
score_list = [knn_score.mean(),naive_bay_score.mean(),dec_tr_score.mean(),rand_for_score.mean(),sgd_score.mean(),lin_svc_score.mean(),log_reg_score.mean()]

for model, n, score in zip(models, np.arange(len(models)), score_list):
    score_table.loc[n,'model'] = model
    score_table.loc[n,'cv_10'] = score 
    
    










