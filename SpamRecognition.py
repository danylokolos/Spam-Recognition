# -*- coding: utf-8 -*-
"""
Created on Sun Nov 28 22:10:56 2021

@author: Danylo
"""

#%% Spam Recognition

#%% Read in Data
import numpy as np
import pandas as pd
import pickle
import os


infile = "InputData/Spam Email raw text for NLP.csv"
df = pd.read_csv(infile)


#%% Data Wrangling
# identify feature and target
feature_names = ['MESSAGE']
target_names = ['CATEGORY']


#%% Preprocessing
# drop columns with useless info
df = df.drop(['FILE_NAME'],axis=1)
df.columns
# Category --> 0 = NOT SPAM, 1 = SPAM

# encode categorical target
from sklearn.preprocessing import LabelEncoder
lenc = LabelEncoder()
_y = df[['CATEGORY']].values
lenc.fit(_y.ravel())
#df[['satisfaction']] = lenc.transform(_y.ravel())
_a = pd.DataFrame(lenc.transform(_y.ravel()))
df[['CATEGORY']] = _a

target_possibilities = df.CATEGORY.unique()
target_possibilities = ['NOT SPAM','SPAM']

# save label encoder
lenc_filename= "lenc_target.pkl"
with open(lenc_filename, 'wb') as outfile:
    pickle.dump(lenc,outfile)




#%% tf-idf
#102542 unique words to 8694 unique words with max_df and min_df
from sklearn.feature_extraction.text import TfidfVectorizer
strip_accents = 'ascii'
lowercase = True
max_df = 0.4
min_df = 10
vectorizer = TfidfVectorizer(strip_accents=strip_accents,
                             lowercase=lowercase,
                             max_df=max_df,
                             min_df=min_df)
X = vectorizer.fit_transform(df['MESSAGE'])

#consider adding NLTK to add stemming functionality

# save tfidf vectorizer
vectorizer_filename= "vectorizer.pkl"
with open(vectorizer_filename, 'wb') as outfile:
    pickle.dump(vectorizer,outfile)



#%% Split data
from sklearn.model_selection import train_test_split
X = X
y = df[target_names]

random_state = 42
test_size = 0.2
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

"""
# Undersample Majority Class - Random - results in lower accuracy
# Accuracy 98.1% with undersampling, better without
from imblearn.under_sampling import RandomUnderSampler
print('Length of X_train before Rebalance:', str(X_train.shape[0]))
undersample = RandomUnderSampler(sampling_strategy='majority')
X_train_under, y_train_under = undersample.fit_resample(X_train, y_train)
print('Length of X_train after Rebalance:', str(X_train_under.shape[0]))

X_train = X_train_under
y_train = y_train_under
"""


#%% Machine Learning Model
from sklearn.ensemble import RandomForestClassifier

n_estimators = 100
random_state = 42
model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
model.fit(X_train,y_train.values.ravel())


# save model
model_filename = "model_RF.pkl"
with open(model_filename, 'wb') as outfile:
    pickle.dump(model,outfile)
    




#%% Run Model on Test Dataset
y_pred = model.predict(X_test)




#%% Analyze Results
# Feature Importance
feature_importance = list(zip(X_train, model.feature_importances_))
print('========== Feature Importance ==========')
print(*feature_importance, sep='\n')
print('========== END ==========')

# Accuracy Score
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
print('========== Accuracy Score ==========')
print(accuracy)
print('========== END ==========')


# Confusion Matrix
from sklearn.metrics import confusion_matrix
conf_matrix = confusion_matrix(y_test, y_pred)

print('========== Confusion Matrix ==========')
print(conf_matrix)
print('========== END ==========')



#%% Plot Confusion Matrix
import matplotlib.pyplot as plt

plt.figure(figsize = (10, 10))
cmap = plt.cm.Blues
plt.imshow(conf_matrix,cmap=cmap)
plt.grid(False)
plt.title('Customer Satisfaction Confusion Matrix', size = 24)
plt.colorbar(aspect=5)
#output_labels = lenc.inverse_transform(target_possibilities)
output_labels = target_possibilities

tick_marks = np.arange(len(output_labels))
plt.xticks(tick_marks,output_labels,rotation=30,fontsize='xx-large')
plt.yticks(tick_marks,output_labels,fontsize='xx-large')
for ii in range(len(output_labels)):
    for jj in range(len(output_labels)):
        if conf_matrix[ii,jj] > np.max(conf_matrix)/2:
            plt.text(ii,jj,conf_matrix[ii,jj],horizontalalignment="center",color="white",fontsize='xx-large')
        else:
            plt.text(ii,jj,conf_matrix[ii,jj],horizontalalignment="center",fontsize='xx-large')
plt.tight_layout(pad=1)
plt.savefig('Plot_ConfusionMatrix.png')



#%% Plot Confusion Matrix Normalized
conf_matrix_norm = conf_matrix / conf_matrix.max()
plt.figure(figsize = (10, 10))
cmap = plt.cm.Blues
plt.imshow(conf_matrix_norm,cmap=cmap)
plt.grid(False)
plt.title('Customer Satisfaction Confusion Matrix Normalized', size = 24)
plt.colorbar(aspect=5)
#output_labels = lenc.inverse_transform(target_possibilities)
output_labels = target_possibilities
tick_marks = np.arange(len(output_labels))
plt.xticks(tick_marks,output_labels,rotation=30,fontsize='xx-large')
plt.yticks(tick_marks,output_labels,fontsize='xx-large')
for ii in range(len(output_labels)):
    for jj in range(len(output_labels)):
        if conf_matrix_norm[ii,jj] > np.max(conf_matrix_norm)/2:
            plt.text(ii,jj,"{:.3f}".format(conf_matrix_norm[ii,jj]),horizontalalignment="center",color="white",fontsize='xx-large')
        else:
            plt.text(ii,jj,"{:.3f}".format(conf_matrix_norm[ii,jj]),horizontalalignment="center",fontsize='xx-large')
plt.tight_layout(pad=1)
plt.savefig('Plot_ConfusionMatrixNorm.png')



#%% Plot Precision-Recall Curve
from sklearn.metrics import PrecisionRecallDisplay

y_score = model.predict_proba(X_test)
y_score = y_score[:,1]
plt.figure()
disp = PrecisionRecallDisplay.from_predictions(y_test, y_score, name="Random Forest")
_ = disp.ax_.set_title('Precision Recall Curve')
plt.savefig('Plot_PrecisionRecallCurve.png')



#%% Plot ROC Curve
import sklearn.metrics as metrics

probs = model.predict_proba(X_test)
preds = probs[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, preds)
roc_auc = metrics.auc(fpr, tpr)

plt.figure()
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.02, 1])
plt.ylim([0, 1.02])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('Plot_ROCCurve.png')


#%% See mispredicted emails


print('SPAM, Predicted as NOT SPAM)
# Predicted not spam
for ii in range(0,len(y_test)):
    if y_pred[ii] - float(y_test.iloc[ii].values)  == -1:
        print(df.values[y_test.index[ii]])
        input()

print('NOT SPAM, Predicted as SPAM)
# Predicted spam        
for ii in range(0,len(y_test)):
    if float(y_test.iloc[ii].values) - y_pred[ii]   == -1:
        print(df.values[y_test.index[ii]])
        input()

