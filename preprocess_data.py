# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 02:02:05 2018

@author: ishan
"""

#----------- Import Libraries ----------------------------#

# for linear Algebra use
import numpy as np

# for Data processing
import pandas as pd
#for data visualization 
from matplotlib import pyplot as plt

#for
from sklearn.preprocessing import LabelEncoder

# Import dataset
df_orig = pd.read_csv('bank-additional.csv')
print(df.describe())

display(df.head())

df.hist(figsize=(15,15), grid=False)
#create copy
df=df_orig.copy()

#label encode categorical data in the Training data set manually

df['job'].replace({
        'unknown' : -0.378390,
        'blue-collar':1,
        'entrepreneur':2,
        'technician':3,
        'admin.':4,
        'retired':5,
        'services':6,
        'self-employed':7,
        'student':8,
        'management':9,
        'unemployed':10,
        'housemaid':11
        },inplace=True)
    
df['marital'].replace({
        'unknown' : -0.378390,
        'single':1,
        'divorced':2,
        'married':3
        },inplace=True)
    

df['education'].replace({
        'unknown' : -0.378390,
        'basic.4y':1,
        'basic.6y':2,
        'basic.9y':3,
        'high.school':4,
        'illiterate':-0.378390,
        'professional.course':6,
        'university.degree':7
        },inplace=True)

df['default'].replace({
        'unknown' : -0.378390,
        'no':0,
        'yes':1
        },inplace=True)

df['housing'].replace({
        'unknown' : -0.378390,
        'no':1,
        'yes':2
        },inplace=True)

df['loan'].replace({
        'unknown' : -0.378390,
        'no':1,
        'yes':2
        },inplace=True)

#Histogram for the number attributes including unknowns in each feature

plt.figure()
plt.hist(df['job'], bins=30, normed=True, alpha=0.9,
         histtype='stepfilled',color='blue',
         edgecolor='none')
plt.show()

plt.figure()
plt.hist(df['marital'], bins=30, normed=True, alpha=0.9,
         histtype='stepfilled',color='blue',
         edgecolor='none')
plt.show()

plt.figure()
plt.hist(df['education'], bins=30, normed=True, alpha=0.9,
         histtype='stepfilled',color='blue',
         edgecolor='none')
plt.show()

plt.figure()
plt.hist(df['default'], bins=30, normed=True, alpha=0.9,
         histtype='stepfilled',color='blue',
         edgecolor='none')
plt.show()

plt.figure()
plt.hist(df['housing'], bins=30, normed=True, alpha=0.9,
         histtype='stepfilled',color='blue',
         edgecolor='none')
plt.show()

plt.figure()
plt.hist(df['loan'], bins=30, normed=True, alpha=0.9,
         histtype='stepfilled',color='blue',
         edgecolor='none')
plt.show()



df['contact'] = df.contact.map({'cellular':0, 'telephone':1})
df_nan_removed = df.replace(-0.378390,np.nan)       
df_nan_removed = df_nan_removed.dropna()
df_nan_removed.drop('default', axis=1, inplace=True)

df_nan_removed['job'].replace({
        1:'blue-collar',
        2:'entrepreneur',
        3:'technician',
        4:'admin.',
        5:'retired',
        6:'services',
        7:'self-employed',
        8:'student',
        9:'management',
        10:'unemployed',
        11:'housemaid'
        },inplace=True)
    
df_nan_removed['marital'].replace({
        1:'single',
        2:'divorced',
        3:'married'
        },inplace=True)

df_nan_removed['education'].replace({
        1:'basic.4y',
        2:'basic.6y',
        3:'basic.9y',
        4:'high.school',
        6:'professional.course',
        7:'university.degree'
        },inplace=True)    
    
df_nan_removed['housing'].replace({
        1:'no',
        2:'yes'
        },inplace=True)

df_nan_removed['loan'].replace({
        1:'no',
        2:'yes'
        },inplace=True)    

# Number of Yes is only 1 , lets treat this as the same class as unknown in df which has unknowns

df['default'].replace({
        -0.378390:1
        },inplace=True)

df['job'].replace({
        -0.378390:'unknown',
        1:'blue-collar',
        2:'entrepreneur',
        3:'technician',
        4:'admin.',
        5:'retired',
        6:'services',
        7:'self-employed',
        8:'student',
        9:'management',
        10:'unemployed',
        11:'housemaid'
        },inplace=True)
    
df['marital'].replace({
        -0.378390:'unknown',
        1:'single',
        2:'divorced',
        3:'married'
        },inplace=True)

df['education'].replace({
        -0.378390:'unknown',
        1:'basic.4y',
        2:'basic.6y',
        3:'basic.9y',
        4:'high.school',
        -0.378390:'illiterate',
        6:'professional.course',
        7:'university.degree'
        },inplace=True)    
    
df['housing'].replace({
        -0.378390:'unknown',
        1:'no',
        2:'yes'
        },inplace=True)

df['loan'].replace({
        -0.378390:'unknown',
        1:'no',
        2:'yes'
        },inplace=True)    

#one hot encode df without nan and also one hot encode df with nan
#with unknowns
data_df = df.iloc[:,:-1]
label_df = df.iloc[:,19]

label_df = label_df.to_frame()
label_df['y'] = label_df.y.map({'no':0, 'yes':1})

label_enc_df = data_df.copy()

label_enc_df['job'].replace({
        'unknown' : 0,
        'blue-collar':1,
        'entrepreneur':2,
        'technician':3,
        'admin.':4,
        'retired':5,
        'services':6,
        'self-employed':7,
        'student':8,
        'management':9,
        'unemployed':10,
        'housemaid':11
        },inplace=True)
    
label_enc_df['marital'].replace({
        'unknown' : 0,
        'single':1,
        'divorced':2,
        'married':3
        },inplace=True)
    

label_enc_df['education'].replace({
        'unknown' : 0,
        'basic.4y':1,
        'basic.6y':2,
        'basic.9y':3,
        'high.school':4,
        'illiterate':0,
        'professional.course':6,
        'university.degree':7
        },inplace=True)


label_enc_df['housing'].replace({
        'unknown' : 0,
        'no':1,
        'yes':2
        },inplace=True)

label_enc_df['loan'].replace({
        'unknown' : 0,
        'no':1,
        'yes':2
        },inplace=True)

label_enc_df['poutcome'].replace({
        'nonexistent' : 0,
        'failure':1,
        'success':2
        },inplace=True)

label_enc_df['month'].replace({
        'may' :0,
        'jun' :1,
        'nov' :2,
        'sep' :3,
        'jul' :4,
        'aug' :5,
        'mar' :6,
        'oct' :7,
        'apr' :8,
        'dec' :9
        },inplace=True)

label_enc_df['day_of_week'].replace({
       'fri' :0,
       'wed' :1,
       'mon' :2,
       'thu' :3,
       'tue':4,
        },inplace=True)


#One more dataset for doing training on - label encoded with unknown
label_pre_unknown_df = pd.concat([label_enc_df,label_df], axis=1)
label_pre_unknown_df.to_csv('label_enc_preprocessed_with_unknown.csv', encoding='utf-8', index=False)

data_df = pd.get_dummies(data_df,drop_first=False)


#Data preprocess set with unknown values treated as an independent attribute
pre_unknown_df = pd.concat([data_df,label_df], axis=1)
pre_unknown_df.to_csv('preprocessed_with_unknown.csv', encoding='utf-8', index=False)

#without unknowns and removal of 'default' column feature + removal of illiterate row for education

nan_removed_data_df = df_nan_removed.iloc[:,:-1]
nan_removed_label_df = df_nan_removed.iloc[:,18]

nan_removed_label_df = nan_removed_label_df.to_frame()
nan_removed_label_df['y'] = nan_removed_label_df.y.map({'no':0, 'yes':1})

label_enc_nan_removed_df = nan_removed_data_df.copy()

label_enc_nan_removed_df['job'].replace({
        'blue-collar':1,
        'entrepreneur':2,
        'technician':3,
        'admin.':4,
        'retired':5,
        'services':6,
        'self-employed':7,
        'student':8,
        'management':9,
        'unemployed':10,
        'housemaid':11
        },inplace=True)
    
label_enc_nan_removed_df['marital'].replace({
        'single':1,
        'divorced':2,
        'married':3
        },inplace=True)
    

label_enc_nan_removed_df['education'].replace({
        'basic.4y':1,
        'basic.6y':2,
        'basic.9y':3,
        'high.school':4,
        'professional.course':6,
        'university.degree':7
        },inplace=True)


label_enc_nan_removed_df['housing'].replace({
        'no':1,
        'yes':2
        },inplace=True)

label_enc_nan_removed_df['loan'].replace({
        'no':1,
        'yes':2
        },inplace=True)

label_enc_nan_removed_df['month'].replace({
        'may' :0,
        'jun' :1,
        'nov' :2,
        'sep' :3,
        'jul' :4,
        'aug' :5,
        'mar' :6,
        'oct' :7,
        'apr' :8,
        'dec' :9
        },inplace=True)

label_enc_nan_removed_df['day_of_week'].replace({
       'fri' :0,
       'wed' :1,
       'mon' :2,
       'thu' :3,
       'tue':4,
        },inplace=True)

label_enc_nan_removed_df['poutcome'].replace({
        'nonexistent' : 0,
        'failure':1,
        'success':2
        },inplace=True)

#One more dataset for doing training on - label encoded without unknown
label_pre_without_unknown_df = pd.concat([label_enc_nan_removed_df,nan_removed_label_df], axis=1)
label_pre_without_unknown_df.to_csv('label_enc_preprocessed_without_unknown.csv', encoding='utf-8', index=False)

nan_removed_data_df = pd.get_dummies(nan_removed_data_df,drop_first=False)

#Data preprocess set with unknown values treated as an independent attribute
pre_without_unknown_df = pd.concat([nan_removed_data_df,nan_removed_label_df], axis=1)
pre_without_unknown_df.to_csv('preprocessed_without_unknown.csv', encoding='utf-8', index=False)


#total 4 datasets to do the operations on
''' 1.'label_enc_preprocessed_with_unknown.csv'
    2.'preprocessed_with_unknown.csv'
    3.'label_enc_preprocessed_without_unknown.csv'
    4.'preprocessed_without_unknown.csv' '''