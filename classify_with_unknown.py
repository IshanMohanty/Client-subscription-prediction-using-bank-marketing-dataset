# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 04:38:44 2018

@author: ishan
"""

# for linear Algebra use
import numpy as np

# for Data processing
import pandas as pd

#for data visualization 
from matplotlib import pyplot as plt

#feature scaling library
from sklearn.preprocessing import MinMaxScaler,RobustScaler

#Library for splitting training and testing data
from sklearn.model_selection import train_test_split

#import random forest library
from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_selection import SelectFromModel

from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report

from sklearn.metrics import roc_auc_score



#from sklearn.linear_model import SGDClassifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.tree import DecisionTreeClassifier


from sklearn.model_selection import cross_val_score



from sklearn import metrics


# Import datasets

#onehotencoded data with unknown

df_with_unknown = pd.read_csv('preprocessed_with_unknown.csv')
data_df_WU = df_with_unknown.iloc[:,:-1]
label_df_WU = df_with_unknown.iloc[:,58]



#Split into Train and test for both cases

train_data_df_WU ,test_data_df_WU ,train_label_df_WU ,test_label_df_WU = train_test_split(data_df_WU,label_df_WU, test_size=0.10 , random_state=40)


# Feature scaling
# What to do about Normalizing negative data ? Any significance
# Feature engineering like boxplot and handling negative data and outlier detection ?

norm_scale_1 = MinMaxScaler(feature_range = (0,1))
train_data_df_WU = norm_scale_1.fit_transform(train_data_df_WU)
test_data_df_WU = norm_scale_1.transform(test_data_df_WU)

#norm_scale_1 = RobustScaler()
#train_data_df_WU = norm_scale_1.fit_transform(train_data_df_WU)
#test_data_df_WU = norm_scale_1.transform(test_data_df_WU)


#feature engineering and checking feature importance

feature_names = ['age', 'default', 'contact', 'campaign', 'pdays', 'previous', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed', 'job_admin.', 'job_blue-collar', 'job_entrepreneur', 'job_housemaid', 'job_management', 'job_retired', 'job_self-employed', 'job_services', 'job_student', 'job_technician', 'job_unemployed', 'job_unknown', 'marital_divorced', 'marital_married', 'marital_single', 'marital_unknown', 'education_basic.4y', 'education_basic.6y', 'education_basic.9y', 'education_high.school', 'education_illiterate', 'education_professional.course', 'education_university.degree', 'housing_no', 'housing_unknown', 'housing_yes', 'loan_no', 'loan_unknown', 'loan_yes', 'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 'month_may', 'month_nov', 'month_oct', 'month_sep', 'day_of_week_fri', 'day_of_week_mon', 'day_of_week_thu', 'day_of_week_tue', 'day_of_week_wed', 'poutcome_failure', 'poutcome_nonexistent', 'poutcome_success']
clf_rfc = RandomForestClassifier(n_estimators=100, random_state=0, n_jobs=-1)

# Train classifier
clf_rfc.fit(train_data_df_WU, train_label_df_WU)

feats = {}
# Print the name and gini importance of each feature
for feature in zip(feature_names, clf_rfc.feature_importances_):
    print(feature)
   

for feature,importance_value in zip(feature_names, clf_rfc.feature_importances_):
    feats[feature] = importance_value
     
importances = pd.DataFrame.from_dict(feats, orient='index').rename(columns={0: 'Gini-importance'})
plt.figure()
importances.sort_values(by='Gini-importance').plot(kind='bar', rot=45)    


# Create a selector object that will use the random forest classifier to identify
# features that have an importance of more than 0.15
sfm = SelectFromModel(clf_rfc, threshold=0.01)

# Train the selector
sfm.fit(train_data_df_WU, train_label_df_WU)

print("\n")
print("\n")
print("\n")


# Print the names of the most important features
for feature_list_index in sfm.get_support(indices=True):
    print(feature_names[feature_list_index])
    
        
# Transform the data to create a new dataset containing only the most important features
# Note: We have to apply the transform to both the training X and test X data.
X_important_train = sfm.transform(train_data_df_WU)
X_important_test = sfm.transform(test_data_df_WU)    


#######################################################################################################


#Classification using Decision Tree

print("\n")
print("\n")
print("Using Decision Tree for Classification")
print("\n")
#train on full data
clf_tree = DecisionTreeClassifier()
clf_tree.fit(train_data_df_WU, train_label_df_WU)

# Apply The Full Featured Classifier To The Test Data
y_pred_tree = clf_tree.predict(test_data_df_WU)

print("\n")
print("Accuracy of full feature model: ")
print("\n")

# View The Accuracy Of Our Full Feature Model
print(accuracy_score(test_label_df_WU, y_pred_tree))
print("\n")
# Model Prediction F1 score -- Precision , Recall
print(classification_report(test_label_df_WU,y_pred_tree))
np.sum(y_pred_tree)

tree_roc_auc = roc_auc_score(test_label_df_WU, y_pred_tree)

print("ROC AUC Score : ", tree_roc_auc)


print("\n")
print("\n")

#Use for important features

clf_tree_imp = DecisionTreeClassifier()

# Train the new classifier on the new dataset containing the most important features
clf_tree_imp.fit(X_important_train, train_label_df_WU)


# Apply The Important Classifier To The Test Data
y_important_pred_clf_tree_imp = clf_tree_imp.predict(X_important_test)

#----- limited feature Model -----------------------#

print("Accuracy of limited model: ")
# View The Accuracy Of Our Limited Feature Model
print(accuracy_score(test_label_df_WU, y_important_pred_clf_tree_imp))
print("\n")
# Model Prediction F1 score -- Precision , Recall
print(classification_report(test_label_df_WU,y_important_pred_clf_tree_imp))
np.sum(y_important_pred_clf_tree_imp)

clf_tree_imp_roc_auc = roc_auc_score(test_label_df_WU, y_important_pred_clf_tree_imp)

print("ROC AUC Score : ", clf_tree_imp_roc_auc)
print(" ")
print(" ")
print(" ")
print(" ")



##########################################################################################################3



#Classification using SVM



print("Using SVM for Classification")
print(" ")
#train on full data
clf_SVM = SVC(kernel="rbf")
clf_SVM.fit(train_data_df_WU, train_label_df_WU)

# Apply The Full Featured Classifier To The Test Data
y_pred_SVM = clf_SVM.predict(test_data_df_WU)

results = cross_val_score(clf_SVM, train_data_df_WU, train_label_df_WU, cv=5)
print("mean cross-validated SVM training accuracy", results.mean())


print("\n")
print("Accuracy of full feature model: ")
print("\n")

# View The Accuracy Of Our Full Feature Model
print(accuracy_score(test_label_df_WU, y_pred_SVM))
print("\n")
# Model Prediction F1 score -- Precision , Recall
print(classification_report(test_label_df_WU,y_pred_SVM))
np.sum(y_pred_SVM)

SVM_roc_auc = roc_auc_score(test_label_df_WU, y_pred_SVM)

print("ROC AUC Score : ", SVM_roc_auc)


print("\n")
print("\n")

#Use for important features

clf_SVM_imp = SVC(kernel="rbf",probability=True)

# Train the new classifier on the new dataset containing the most important features
clf_SVM_imp.fit(X_important_train, train_label_df_WU)


# Apply The Important Classifier To The Test Data
y_important_pred_clf_SVM_imp = clf_SVM_imp.predict(X_important_test)

results_imp = cross_val_score(clf_SVM_imp, X_important_train, train_label_df_WU, cv=5)
print("mean cross-validated SVM training accuracy for limited data model ", results_imp.mean())

#----- limited feature Model -----------------------#

print("Accuracy of limited model: ")
# View The Accuracy Of Our Limited Feature Model
print(accuracy_score(test_label_df_WU, y_important_pred_clf_SVM_imp))
print("\n")
# Model Prediction F1 score -- Precision , Recall
print(classification_report(test_label_df_WU,y_important_pred_clf_SVM_imp))
np.sum(y_important_pred_clf_SVM_imp)

clf_SVM_imp_roc_auc = roc_auc_score(test_label_df_WU, y_important_pred_clf_SVM_imp)

print("ROC AUC Score : ", clf_SVM_imp_roc_auc)
print(" ")
print(" ")
print(" ")
print(" ")






##############################################################################################################


#Classification for Logistic regression
print("Using Logistic regression for Classification")
print(" ")
#train on full data
clf_log = LogisticRegression()
clf_log.fit(train_data_df_WU, train_label_df_WU)

# Apply The Full Featured Classifier To The Test Data
y_pred_log = clf_log.predict(test_data_df_WU)

print("\n")
print("Accuracy of full feature model: ")
print("\n")

# View The Accuracy Of Our Full Feature Model
print(accuracy_score(test_label_df_WU, y_pred_log))
print("\n")
# Model Prediction F1 score -- Precision , Recall
print(classification_report(test_label_df_WU,y_pred_log))
np.sum(y_pred_log)

log_roc_auc = roc_auc_score(test_label_df_WU, y_pred_log)

print("ROC AUC Score : ", log_roc_auc)


print("\n")
print("\n")

#Use for important features

clf_log_imp = LogisticRegression()

# Train the new classifier on the new dataset containing the most important features
clf_log_imp.fit(X_important_train, train_label_df_WU)


# Apply The Important Classifier To The Test Data
y_important_pred_clf_log_imp = clf_log_imp.predict(X_important_test)

#----- limited feature Model -----------------------#

print("Accuracy of limited model: ")
# View The Accuracy Of Our Limited Feature Model
print(accuracy_score(test_label_df_WU, y_important_pred_clf_log_imp))
print("\n")
# Model Prediction F1 score -- Precision , Recall
print(classification_report(test_label_df_WU,y_important_pred_clf_log_imp))
np.sum(y_important_pred_clf_log_imp)

clf_log_imp_roc_auc = roc_auc_score(test_label_df_WU, y_important_pred_clf_log_imp)

print("ROC AUC Score : ", clf_log_imp_roc_auc)
print(" ")
print(" ")
print(" ")
print(" ")




############################################################################################################

#Classification for KNN
print("Using KNN Classification")

print(" ")
print(" ")
#train on full data
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(train_data_df_WU, train_label_df_WU)

# Apply The Full Featured Classifier To The Test Data
y_pred_knn = knn.predict(test_data_df_WU)

print(" ")
print(" ")
print("Accuracy of full feature model: ")

# View The Accuracy Of Our Full Feature Model
print(accuracy_score(test_label_df_WU, y_pred_knn))

# Model Prediction F1 score -- Precision , Recall
print(classification_report(test_label_df_WU,y_pred_knn))
np.sum(y_pred_knn)

knn_roc_auc = roc_auc_score(test_label_df_WU, y_pred_knn)

print("ROC AUC Score : ", knn_roc_auc)

#Use for important features

knn_imp = KNeighborsClassifier(n_neighbors = 3)

# Train the new classifier on the new dataset containing the most important features
knn_imp.fit(X_important_train, train_label_df_WU)


# Apply The Important Classifier To The Test Data
y_important_pred_knn_imp = knn_imp.predict(X_important_test)

#----- limited feature Model -----------------------#

print("Accuracy of limited model: ")
# View The Accuracy Of Our Limited Feature Model
print(accuracy_score(test_label_df_WU, y_important_pred_knn_imp))

# Model Prediction F1 score -- Precision , Recall
print(classification_report(test_label_df_WU,y_important_pred_knn_imp))
np.sum(y_important_pred_knn_imp)

knn_imp_roc_auc = roc_auc_score(test_label_df_WU, y_important_pred_knn_imp)

print("ROC AUC Score : ", knn_imp_roc_auc)
print(" ")
print(" ")
print(" ")
print(" ")




########################################################################################################

#Classification for Naive Bayes
print("Using Naive Bayes Classification")

print(" ")
print(" ")
#train on full data
NB = GaussianNB() 
NB.fit(train_data_df_WU, train_label_df_WU)

# Apply The Full Featured Classifier To The Test Data
y_pred_NB = NB.predict(test_data_df_WU)

print(" ")
print(" ")
print("Accuracy of full feature model: ")

# View The Accuracy Of Our Full Feature Model
print(accuracy_score(test_label_df_WU, y_pred_NB))

# Model Prediction F1 score -- Precision , Recall
print(classification_report(test_label_df_WU,y_pred_NB))
np.sum(y_pred_NB)

NB_roc_auc = roc_auc_score(test_label_df_WU, y_pred_NB)

print("ROC AUC Score : ", NB_roc_auc)

#Use for important features

NB_imp = GaussianNB() 

# Train the new classifier on the new dataset containing the most important features
NB_imp.fit(X_important_train, train_label_df_WU)


# Apply The Important Classifier To The Test Data
y_important_pred_NB_imp = NB_imp.predict(X_important_test)

#----- limited feature Model -----------------------#

print("Accuracy of limited model: ")
# View The Accuracy Of Our Limited Feature Model
print(accuracy_score(test_label_df_WU, y_important_pred_NB_imp))

# Model Prediction F1 score -- Precision , Recall
print(classification_report(test_label_df_WU,y_important_pred_NB_imp))
np.sum(y_important_pred_NB_imp)

NB_imp_roc_auc = roc_auc_score(test_label_df_WU, y_important_pred_NB_imp)

print("ROC AUC Score : ", NB_imp_roc_auc)
print(" ")
print(" ")
print(" ")
print(" ")

plt.figure()
y_pred_proba = NB_imp.predict_proba(X_important_test)[:,1]
fpr, tpr, _ = metrics.roc_curve(test_label_df_WU , y_pred_proba)

roc_imp_rf = metrics.roc_auc_score(test_label_df_WU, y_pred_proba)

    
plt.plot([0,1],[0,1],color='red',lw=2,linestyle='--')
plt.plot(fpr,tpr,label="data 1, roc="+str(roc_imp_rf))
plt.legend(loc=4)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC-Curve')
plt.show()


#######################################################################################################


#Classification for Random Forest

print("\n")
print("\n")
print("Using Random forest Classification")

print(" ")
print(" ")
print("Accuracy of full feature model: ")
# Create a new random forest classifier for the most important features
clf_important = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

# Train the new classifier on the new dataset containing the most important features
clf_important.fit(X_important_train, train_label_df_WU)


clf_new_full_rfc = RandomForestClassifier(n_estimators=10000, random_state=0, n_jobs=-1)

# Train classifier
clf_new_full_rfc.fit(train_data_df_WU, train_label_df_WU)

# Apply The Full Featured Classifier To The Test Data
y_pred_new_full_rfc = clf_new_full_rfc.predict(test_data_df_WU)
# View The Accuracy Of Our Full Feature Model
print(accuracy_score(test_label_df_WU, y_pred_new_full_rfc))

# Model Prediction F1 score -- Precision , Recall
print(classification_report(test_label_df_WU,y_pred_new_full_rfc))
np.sum(y_pred_new_full_rfc)

rand_new_full_forest_roc_auc = roc_auc_score(test_label_df_WU, y_pred_new_full_rfc)

print("ROC AUC Score : ", rand_new_full_forest_roc_auc)

# Apply The Important feature Classifier To The Test Data
y_important_pred = clf_important.predict(X_important_test)

#----- limited feature Model -----------------------#

print("Accuracy of limited model: ")
# View The Accuracy Of Our Limited Feature Model
print(accuracy_score(test_label_df_WU, y_important_pred))

# Model Prediction F1 score -- Precision , Recall
print(classification_report(test_label_df_WU,y_important_pred))
np.sum(y_important_pred)

rand_imp_forest_roc_auc = roc_auc_score(test_label_df_WU, y_important_pred)

print("ROC AUC Score : ", rand_imp_forest_roc_auc)







#########################################################################################################















