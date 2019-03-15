# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 16:44:53 2018

@author: Ishan Mohanty

"""

#----------- Import Libraries ----------------------------#

# for linear Algebra use
import numpy as np

# for Data processing
import pandas as pd

#for data visualization 
from matplotlib import pyplot as plt

#Library for splitting training and testing data
from sklearn.cross_validation import train_test_split

#label encoding
from sklearn.preprocessing import LabelEncoder

from sklearn.preprocessing import MinMaxScaler


# Import dataset
df = pd.read_csv('bank-additional.csv')
data_df = df.iloc[:,:-1]
label_df = df.iloc[:,19]



#Split into Train and test 

train_data_df,test_data_df,train_label_df,test_label_df = train_test_split(data_df,label_df, test_size=0.10 , random_state=40)

#Series to frame

train_label_df = train_label_df.to_frame()
test_label_df = test_label_df.to_frame()


#label encode the train and test labels

le = LabelEncoder()
train_label_df['y'] = le.fit_transform(train_label_df['y'])
test_label_df['y']  = le.fit_transform(test_label_df['y'])

#Concatenate the Train data and label together and also the test data and label respectively into a dataframe
# and store into csv file --- Mostly for viewing purpose only

final_train_set = pd.concat([train_data_df,train_label_df], axis=1)
final_test_set = pd.concat([test_data_df,test_label_df], axis=1)
final_train_set.to_csv('bank_train_set.csv', encoding='utf-8', index=False)
final_test_set.to_csv('bank_test_set.csv',encoding='utf-8', index=False)

#take a copy of the test data set and perform some operations according to pre-processed data

test_1_no_unknown = test_data_df.copy()
test_2_unknown = test_data_df.copy()

test_2_unknown = pd.get_dummies(test_2_unknown,drop_first=False)


#Number of rows that have unknown

row_count = train_data_df.shape[0]
col_count = train_data_df.shape[1]

cnt = 0

for i in range(0, row_count):
    if 'unknown' in list(train_data_df.iloc[i,:]):
        cnt = cnt +1

print ("Number rows that have atleast One unknown value are :", cnt)
print (" ")


col = list(train_data_df['job'])
print("Number of unknown values present in the job feature is : ", col.count('unknown'))
print(" ")

col = list(train_data_df['marital'])
print("Number of unknown values present in the marital feature is : ", col.count('unknown'))
print(" ")

col = list(train_data_df['education'])
print("Number of unknown values present in the education feature is : ", col.count('unknown'))
print(" ")

col = list(train_data_df['default'])
print("Number of unknown values present in the default feature is : ", col.count('unknown'))
print(" ")

col = list(train_data_df['housing'])
print("Number of unknown values present in the housing feature is : ", col.count('unknown'))
print(" ")

col = list(train_data_df['loan'])
print("Number of unknown values present in the loan feature is : ", col.count('unknown'))
print(" ")
   

#label encode categorical data in the Training data set manually

train_data_df['job'].replace({
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
    
train_data_df['marital'].replace({
        'unknown' : -0.378390,
        'single':1,
        'divorced':2,
        'married':3
        },inplace=True)
    

train_data_df['education'].replace({
        'unknown' : -0.378390,
        'basic.4y':1,
        'basic.6y':2,
        'basic.9y':3,
        'high.school':4,
        'illiterate':5,
        'professional.course':6,
        'university.degree':7
        },inplace=True)

train_data_df['default'].replace({
        'unknown' : -0.378390,
        'no':0,
        'yes':1
        },inplace=True)

train_data_df['housing'].replace({
        'unknown' : -0.378390,
        'no':1,
        'yes':2
        },inplace=True)

train_data_df['loan'].replace({
        'unknown' : -0.378390,
        'no':1,
        'yes':2
        },inplace=True)
  

#Histogram for the number attributes including unknowns in each feature

plt.hist(train_data_df['job'], bins=30, normed=True, alpha=0.9,
         histtype='stepfilled',color='blue',
         edgecolor='none')
plt.show()

plt.hist(train_data_df['marital'], bins=30, normed=True, alpha=0.9,
         histtype='stepfilled',color='blue',
         edgecolor='none')
plt.show()

plt.hist(train_data_df['education'], bins=30, normed=True, alpha=0.9,
         histtype='stepfilled',color='blue',
         edgecolor='none')
plt.show()

plt.hist(train_data_df['default'], bins=30, normed=True, alpha=0.9,
         histtype='stepfilled',color='blue',
         edgecolor='none')
plt.show()

plt.hist(train_data_df['housing'], bins=30, normed=True, alpha=0.9,
         histtype='stepfilled',color='blue',
         edgecolor='none')
plt.show()

plt.hist(train_data_df['loan'], bins=30, normed=True, alpha=0.9,
         histtype='stepfilled',color='blue',
         edgecolor='none')
plt.show()


#Keep a copy of the bank data set

train_enc_data =  train_data_df.copy(deep=True) #Use this for further processes

#Remove Unknowns which are encoded as Zeros

#for i in range(0, row_count):
#    if 0 in list(train_data_df.iloc[i,1:6]):

train_data_df.replace(-0.378390,np.nan,inplace=True)    

# Use this for training        
nan_dropped_train_data_df = train_data_df.dropna() #use this data for training
nan_dropped_train_data_df.to_csv('removed_nan_preprocess_data_set_1.csv',encoding='utf-8', index=False)
        
## looks like a useful feature

df_modify_label = df.copy()
df_modify_label['y'] = df_modify_label.y.map({'no':0, 'yes':1})
display(df_modify_label.groupby('default').y.mean())
df_modify_label.default.value_counts()

mod_train_data_df = train_data_df.copy() 
mod_train_data_df['default'].replace(np.nan,1,inplace=True)
mod_train_data_df['contact'] = train_data_df.contact.map({'cellular':0, 'telephone':1})


mod_train_data_df['job'].replace({
        np.nan:'unknown',
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
    
mod_train_data_df['marital'].replace({
        np.nan:'unknown',
        1:'single',
        2:'divorced',
        3:'married'
        },inplace=True)

mod_train_data_df['education'].replace({
        np.nan:'unknown',
        1:'basic.4y',
        2:'basic.6y',
        3:'basic.9y',
        4:'high.school',
        5:'illiterate',
        6:'professional.course',
        7:'university.degree'
        },inplace=True)    
    
mod_train_data_df['housing'].replace({
        np.nan:'unknown',
        1:'no',
        2:'yes'
        },inplace=True)

mod_train_data_df['loan'].replace({
        np.nan:'unknown',
        1:'no',
        2:'yes'
        },inplace=True)    

# one hot code job_dummy
job_dummy = pd.get_dummies(mod_train_data_df['job'],prefix='job',drop_first=False)

# onehotcode poutcome_dummies
poutcome_dummy = pd.get_dummies(mod_train_data_df.poutcome,prefix='poutcome',drop_first=False)

data_clean_df = pd.concat([mod_train_data_df, job_dummy, poutcome_dummy], axis=1)
data_clean_df.drop('job', axis=1, inplace=True)
data_clean_df.drop('poutcome', axis=1, inplace=True)
data_clean_df.replace(np.nan,0,inplace=True)

month_dummy = pd.get_dummies(data_clean_df['month'],prefix='month',drop_first=False)
day_of_week_dummy = pd.get_dummies(data_clean_df['day_of_week'],prefix='data_of_week',drop_first=False)
marital_dummy = pd.get_dummies(data_clean_df['marital'],prefix='marital',drop_first=False)
education_dummy = pd.get_dummies(data_clean_df['education'],prefix='education',drop_first=False)
housing_dummy = pd.get_dummies(data_clean_df['housing'],prefix='housing',drop_first=False)
loan_dummy = pd.get_dummies(data_clean_df['loan'],prefix='loan',drop_first=False)

data_clean_df = pd.concat([data_clean_df, month_dummy, day_of_week_dummy,marital_dummy,education_dummy,housing_dummy,loan_dummy], axis=1)
data_clean_df.drop('marital', axis=1, inplace=True)
data_clean_df.drop('education', axis=1, inplace=True)
data_clean_df.drop('housing', axis=1, inplace=True)
data_clean_df.drop('loan', axis=1, inplace=True)
data_clean_df.drop('month', axis=1, inplace=True)
data_clean_df.drop('day_of_week', axis=1, inplace=True)


# Standardize and Normalize 

# create scalers
scaler_data_clean_df = MinMaxScaler(feature_range = (0,1))
# fit and transform
data_clean_df['age','campaign','pdays','previous','emp.var.rate','cons.price.idx','cons.conf.idx','euribor3m','nr.employed' ] = scaler_data_clean_df.fit_transform(data_clean_df[['age']])






