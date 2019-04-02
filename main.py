
import statsmodels.formula.api as smf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

df_salary = pd.read_csv('salary.csv')

df_salary.dropna()

data = df_salary['salary']

X_train, X_test, y_train, y_test = train_test_split(df_salary, data, test_size=0.85, random_state=30)


lm = smf.ols(formula='salary ~ yearsworked', data=X_train).fit()

lm.params

df_predictions = pd.DataFrame({'yearsworked': [12]})
lm.predict(df_predictions)


ei = pd.DataFrame({'yearsworked': [80]})
lm.predict(ei)

predictions = lm.predict(X_train.yearsworked)


lm.conf_int()


lm.pvalues

lm.rsquared


lm.summary()


df_salary.corr()

np.sqrt(mean_squared_error(y_train, predictions))


nm = smf.ols(formula='salary ~ yearsworked', data=X_test).fit()
nm.summary()


test_predictions = nm.predict(X_test)

np.sqrt(mean_squared_error(y_test, test_predictions))
