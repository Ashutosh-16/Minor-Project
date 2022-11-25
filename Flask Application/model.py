# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:10:49 2022

@author: HP
"""

import numpy as np
import pandas as pd
import pickle
data = pd.read_csv("C:/Users/HP/Downloads/dataset.csv")
data.sum()
data.isna().sum()
target = data[["Irrigation"]]
type(target)
data.drop("Irrigation",axis = 1,inplace = True)
data.describe()
data.shape
target["Irrigation"].unique()
Y=target['Irrigation']
Y.head(5)
X = data
X.head(2)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score

categorical_features=['CropType']
one_hot= OneHotEncoder()
transformer=ColumnTransformer([('one_hot',one_hot,categorical_features)], remainder='passthrough')
transformed_X=transformer.fit_transform(X)
transformed_X
X
pd.DataFrame(transformed_X)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(transformed_X,Y,test_size = 0.3)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
y_pred3 = lr.predict(x_test)
print(accuracy_score(y_test,y_pred3))
from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier()
model.fit(x_train,y_train)
y_pred4 = model.predict(x_test)
print(accuracy_score(y_test,y_pred4))
#Saving model to disk
pickle.dump(model,open('model.pkl','wb'))




