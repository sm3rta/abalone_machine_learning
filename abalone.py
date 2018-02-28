
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor

b, XGBoostRegressorErrors, gradientBoostingRegressorErrors=[],[],[]

df = pd.read_csv("abalone.data", names=["sex", "length", "diameter", "height", "whole weight", "shucked weight", "viscera weight", "shell weight", "rings"])

#Labeling object column sex
le = LabelEncoder()
le.fit(df.sex)
df.sex = le.transform(df.sex)

#Removing Outliers (old method)
#indices_to_drop=set()
#
#for j in range(1, len(df.columns)):
#    median=df[df.columns[j]].median()
#    indices_to_drop.update(list(df[df[df.columns[j]]<median*0.05].index.values))
#    indices_to_drop.update(list(df[df[df.columns[j]]>median*2.15].index.values))
#
#df.drop(list(indices_to_drop), inplace=True)

#Removing Outliers
for j in range(1, len(df.columns)):
    df = df[np.abs(df[df.columns[j]]-df[df.columns[j]].mean())<=(3*df[df.columns[j]].std())]

x = df.iloc[:, 0:len(df.columns)-1]
y = df.iloc[:, len(df.columns)-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.33, random_state = 42)

gradientBoostingRegressor = GradientBoostingRegressor()
gradientBoostingRegressor.fit(x_train,y_train)
gradientBoostingRegressorResult = gradientBoostingRegressor.predict(x_test)
gradientBoostingRegressorError = np.sqrt(mean_squared_error(y_test,gradientBoostingRegressorResult))
gradientBoostingRegressorErrors.append(gradientBoostingRegressorError)
print(gradientBoostingRegressorError)

XGBoostRegressor = XGBRegressor()
XGBoostRegressor.fit(x_train,y_train)
XGBoostRegressorResult = XGBoostRegressor.predict(x_test)
XGBoostRegressorError = np.sqrt(mean_squared_error(y_test,XGBoostRegressorResult))
XGBoostRegressorErrors.append(XGBoostRegressorError)
print(XGBoostRegressorError)
