#importing required libraries
import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn import metrics
from sklearn.svm import LinearSVC,SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, MultinomialNB, ComplementNB
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, classification_report, roc_auc_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Perceptron, LogisticRegression

#read the csv file in a dataframe

churn_df = pd.read_csv("Customer_churn_raw.csv")
print(churn_df.head())
churn_df=churn_df.rename(columns={"Call  Failure": "call_failure", "Complains": "complains", "Subscription  Length": "subs_len", "Charge  Amount": "charge_amount",
                   "Seconds of Use": "seconds_of_use", "Frequency of use": "freq_of_use", "Frequency of SMS": "freq_of_sms", "Distinct Called Numbers": "distinct_call_nums",
                   "Age Group": "age_group", "Tariff Plan": "tariff_plan", "Status": "status", "Age": "age", "Customer Value": "customer_value"})

cols_mustafa = churn_df[['call_failure', 'complains', 'subs_len', 'charge_amount']]
#CALL FAILURE
print(cols_mustafa['call_failure'].value_counts())
cols_mustafa['call_failure'] = pd.to_numeric(cols_mustafa['call_failure'].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)
cols_mustafa["call_failure"].replace(np.nan, 1, inplace=True)
print(cols_mustafa['call_failure'].value_counts())

#COMPLAINS
print(cols_mustafa['complains'].value_counts())
cols_mustafa['complains'] = pd.to_numeric(cols_mustafa['complains'].astype(str), errors='coerce').fillna(0).astype(int)
print(cols_mustafa['complains'].value_counts())
cols_mustafa.loc[cols_mustafa['complains'] != 0, 'complains'] = 1
print(cols_mustafa['complains'].value_counts())

#SUBSCRIPTION LENGTH
print(cols_mustafa['subs_len'].value_counts())
cols_mustafa['subs_len'] = cols_mustafa['subs_len'].abs()
cols_mustafa['subs_len'].fillna(
value=cols_mustafa['subs_len'].mean(), inplace=True)
cols_mustafa['subs_len'] = cols_mustafa['subs_len'].astype(int)
print(cols_mustafa['subs_len'].value_counts())

#CHARGE AMOUNT
print(cols_mustafa['charge_amount'].value_counts())
cols_mustafa['charge_amount'].replace(np.nan, 0, inplace=True)
cols_mustafa['charge_amount'] = pd.to_numeric(cols_mustafa['charge_amount'].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)
cols_mustafa['charge_amount'] = cols_mustafa['charge_amount'].abs()
print(cols_mustafa['charge_amount'].value_counts())

churn_df = churn_df.drop(['call_failure', 'complains', 'subs_len', 'charge_amount'],axis = 1)
churn_df = pd.concat([cols_mustafa,churn_df],axis = 1)

#printing the columns
# print(churn_df.columns)
# print("-"*80)
cols_4 = churn_df[['seconds_of_use']]
# print(cols_4_8.info())
# print("-"*80)
# print(cols_4_8.describe())
# print("-"*80)
# print(cols_4_8.isnull().sum())
# print("-"*80)
#print(cols_4_8['seconds_of_use'].value_counts())
print("-"*80)
cols_4["seconds_of_use"].replace(np.nan, 0, inplace=True )
cols_4['seconds_of_use'].replace(['O3915'], 3915, inplace=True )
cols_4['seconds_of_use'] = cols_4['seconds_of_use'].astype(int)
cols_4['seconds_of_use'] = cols_4['seconds_of_use'].abs()
print(cols_4['seconds_of_use'].value_counts())
#checking for null values
print(cols_4.isnull().sum())

churn_df = churn_df.drop(['seconds_of_use'],axis = 1)
churn_df = pd.concat([cols_4,churn_df],axis = 1)


cols_567 = churn_df.iloc[:,[5,6,7]]
for x in cols_567.columns:
    mean_column = np.nanmean(churn_df[x].values)
    churn_df[x].replace(np.nan, mean_column, inplace=True)
    churn_df[x] = abs(churn_df[x])


print(churn_df['age'].value_counts())
churn_df.insert(11, "ageGroup", 0)
df_mean = round(churn_df["age"].mean())

for i in range(len(churn_df)):
    if(churn_df.loc[i,'age'] < 0) | (churn_df.loc[i,'age'] > 100) :
        churn_df.loc[i,'age'] = df_mean
for i in range(len(churn_df)):
    if(churn_df.loc[i,'age'] > 0) & (churn_df.loc[i,'age']<=15) :
        churn_df.loc[i,'ageGroup'] = 1
    if(churn_df.loc[i,'age'] > 15) & (churn_df.loc[i,'age']<=30) :
        churn_df.loc[i,'ageGroup'] = 2
    if(churn_df.loc[i,'age'] > 30) & (churn_df.loc[i,'age']<=45) :
        churn_df.loc[i,'ageGroup'] = 3
    if(churn_df.loc[i,'age'] > 45) & (churn_df.loc[i,'age']<=60) :
        churn_df.loc[i,'ageGroup'] = 4
    if(churn_df.loc[i,'age'] > 60) & (churn_df.loc[i,'age']<=80) :
        churn_df.loc[i,'ageGroup'] = 5
    if(churn_df.loc[i,'age'] > 80) & (churn_df.loc[i,'age']< 0) :
        churn_df.loc[i,'ageGroup'] = 2

print(churn_df['ageGroup'].value_counts())
print('*'*50)
print(churn_df.dtypes)
print('*'*50)

list_outliers = ['customer_value', 'freq_of_use', 'freq_of_sms', 'distinct_call_nums', 'FN', 'FP']

for column in list_outliers:
    col_out_indx = np.where(
        churn_df[column] < (churn_df[column].mean() + (4 * churn_df[column].std())))
    for outlier in col_out_indx:
        churn_df[column][outlier] = churn_df[column].median()


cols_3 = churn_df[['status', 'age', 'customer_value', 'FN', 'FP']]

cols_3['customer_value'] = pd.to_numeric(cols_3['customer_value'].astype(str).str.replace(',',''),
                                                     errors='coerce').fillna(0).astype(int)

cols_3["customer_value"].replace(np.nan, 0, inplace=True )

#replace all the null values with 1
cols_3["customer_value"].replace(np.nan, 1, inplace=True )


# Merging these clean columns to the original dataframe

churn_df = churn_df.drop(['status', 'age', 'customer_value', 'FN', 'FP'],axis = 1)
churn_df = pd.concat([cols_3,churn_df],axis = 1)

print(churn_df.isnull().sum())
import seaborn as sn

sn.set(style="whitegrid",font_scale=1)
plt.figure(figsize=(10, 8))
sn.boxplot(data=churn_df)
plt.xticks(rotation=80)
plt.title("Box plot",fontsize = 20)
plt.tight_layout()
plt.show()

churn_df.to_csv('churn_data_cleaned.csv',index = False)

df = pd.read_csv("churn_data_cleaned.csv")

X = df.iloc[:, :15].to_numpy()
y = df.iloc[:, -1].to_numpy()

print(y.shape)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=1234)
clf = svm.SVC()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

svm_accuracy = accuracy_score(y_test, y_pred)
print(f"SVM accuracy score: {svm_accuracy:9.3f}")
svm_precision = precision_score(y_test, y_pred, average='weighted')
print(f"SVM precision score: {svm_precision:9.3f}")
svm_auc = roc_auc_score(y_test, y_pred, average='weighted')
print(f"SVM auc score: {svm_auc:9.3f}")
svm_recall = recall_score(y_test, y_pred, average='weighted')
print(f"SVM recall score: {svm_recall:9.3f}")
svm_f1 = f1_score(y_test, y_pred, average='weighted')
print(f"SVM f1 score: {svm_f1:9.3f}")



X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y)
clf = LogisticRegression(C=1e10, solver='liblinear')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
logistic_accuracy = accuracy_score(y_test, y_pred)
logistic_precision = precision_score(y_test, y_pred, average='weighted')
logistic_recall = recall_score(y_test, y_pred, average='weighted')
logistic_auc = roc_auc_score(y_test, y_pred, average='weighted')
logistic_f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Logistic accuracy score: {logistic_accuracy:9.3f}")
print(f"Logistic precision score: {logistic_precision:9.3f}")
print(f"Logistic recall score: {logistic_recall:9.3f}")
print(f"Logistic auc score: {logistic_auc:9.3f}")
print(f"Logistic f1 score: {logistic_f1:9.3f}")

from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    f1_score,
    precision_score,
    roc_auc_score,
    accuracy_score,
    recall_score,
)


df = pd.read_csv("churn_data_cleaned.csv")

X = df.iloc[:, :15].to_numpy()
y = df.iloc[:, -1].to_numpy()

print(y.shape)
print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

score = accuracy_score(y_test, y_pred)

print(score)

dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)
y_pred = dt_clf.predict(X_test)
dt_accuracy = accuracy_score(y_test, y_pred)
dt_precision = precision_score(y_test, y_pred, average="weighted")
dt_recall = recall_score(y_test, y_pred, average="weighted")
dt_auc = roc_auc_score(y_test, y_pred, average="weighted")


print("Accuracy for DecisionTreeClassifier: ", dt_accuracy)
print("F1 Score: %.3f" % f1_score(y_test, y_pred, average="weighted"))
print("Precision score: ", dt_precision)
print("Recall score: ", dt_recall)
print("Auc score: ", dt_auc)
