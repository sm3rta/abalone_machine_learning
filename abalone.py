
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor

print("\n")
df = pd.read_csv("abalone.data", names=["sex", "length", "diameter", "height", "whole weight", "shucked weight", "viscera weight", "shell weight", "rings"])
df.drop(["sex"],axis=1)

le = LabelEncoder()
le.fit(df.sex)
df.sex = le.transform(df.sex)

#Removing Outliers
indices_to_drop=set()

for j in range(1, len(df.columns)-1):
    mean=df[df.columns[j]].median()
    indices_to_drop.update(list(df[df[df.columns[j]]<mean*0.015].index.values))
    indices_to_drop.update(list(df[df[df.columns[j]]>mean*2].index.values))

df.drop(list(indices_to_drop), inplace=True)
#----------------

x = df.iloc[:, 0:len(df.columns)-1]
y = df.iloc[:, len(df.columns)-1]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 700)

gradientBoostingRegressor = GradientBoostingRegressor()
gradientBoostingRegressor.fit(x_train,y_train)
gradientBoostingRegressorResult = gradientBoostingRegressor.predict(x_test)
gradientBoostingRegressorError = np.sqrt(mean_squared_error(y_test,gradientBoostingRegressorResult))
print(gradientBoostingRegressorError)

XGBoostRegressor = XGBRegressor()
XGBoostRegressor.fit(x_train,y_train)
XGBoostRegressorResult = XGBoostRegressor.predict(x_test)
XGBoostRegressorError = np.sqrt(mean_squared_error(y_test,XGBoostRegressorResult))
print(XGBoostRegressorError)
