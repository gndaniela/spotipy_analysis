# -*- coding: utf-8 -*-
"""
Created on Tue Nov  2 08:46:39 2021

@author: dgnistor
"""

#%% Libraries call

import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from pandas import json_normalize
from matplotlib import pyplot as plt
from classes import MoodSearch

#%% Create Moods DF to train model

moods_list = ['Sad','Happy','Calm','Energetic'] #change moods here to create your own predictive model
moods_dfname = [i + 'Df' for i in moods_list]

objs = [MoodSearch(i) for i in moods_list]
    
moodsdf = pd.DataFrame()
for obj in objs:
    df = obj.fetch_full_df()
    moodsdf = pd.concat([moodsdf,df])

moodsdf.drop(columns=['type','uri','track_href','analysis_url'],axis=1,inplace=True)

#%% Train - Test - Split

from sklearn.model_selection import train_test_split
from collections import Counter

X = moodsdf.loc[:,~moodsdf.columns.isin(['id','Mood'])]
y = moodsdf['Mood']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
print(f"Training target statistics: {Counter(y_train)}")
print(f"Testing target statistics: {Counter(y_test)}")

#%% Resize samples (if needed to balance mood classes)

from imblearn.under_sampling import RandomUnderSampler

under_sampler = RandomUnderSampler(random_state=42)
X_res, y_res = under_sampler.fit_resample(X_train, y_train)
print(f"Training target statistics: {Counter(y_res)}")
print(f"Testing target statistics: {Counter(y_test)}")


#%% Normalize variables and encode classes

from sklearn.preprocessing import MinMaxScaler

#Normalize the features
X_res = MinMaxScaler().fit_transform(X_res)


#%% Random Forest 

from sklearn.ensemble import RandomForestClassifier
#Create a Gaussian Classifier
clf=RandomForestClassifier(n_estimators=100)

#Train the model using the training sets y_pred=clf.predict(X_test)
clf.fit(X_res,y_res)

y_pred=clf.predict(X_test)

#%% Measure Accuracy
from sklearn import metrics

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

#%% Random Forest Built-in Feature Importance (Plot)

sorted_indexes = clf.feature_importances_.argsort() #sort by order
plt.barh(X.columns[sorted_indexes], clf.feature_importances_[sorted_indexes])
plt.xlabel("Random Forest Feature Importance")

# =============================================================================
# 1 - Instrumentalness
# 2 - Loudness
# 3 - Valence
# 4 - Energy
# =============================================================================

#%% Permutation Based Feature Importance

# =============================================================================
# The permutation based importance can be used to overcome drawbacks of default feature importance 
# computed with mean impurity decrease. It is implemented in scikit-learn as permutation_importance method. 
# As arguments it requires trained model (can be any model compatible with scikit-learn API) and validation (test data). 
# This method will randomly shuffle each feature and compute the change in the model’s performance. 
# The features which impact the performance the most are the most important one.
# The permutation importance can be easily computed
# =============================================================================
from sklearn.inspection import permutation_importance
permutation_imp = permutation_importance(clf, X_res, y_res)


sorted_indexes = permutation_imp.importances_mean.argsort()
plt.barh(X.columns[sorted_indexes], permutation_imp.importances_mean[sorted_indexes])
plt.xlabel("Permutation Importance")

# =============================================================================
# 1 - Instrumentalness
# 2 - Valence
# 3 - Energy
# 4 - Loudness
# =============================================================================
#%% Feature Importance Computed with SHAP Values
import shap
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_res)

shap.summary_plot(shap_values, X_res, feature_names=X.columns, plot_type="bar")

# =============================================================================
# 1 - Instrumentalness
# 2 - Loudness
# 3 - Valence
# 4 - Energy
# =============================================================================

#%% Save model for future usage

import joblib
# save
joblib.dump(clf, r'C:\Users\yourpath\random_forestnpop.joblib')

# Change model path in classes SearchAndPredictTrack() to use your own







