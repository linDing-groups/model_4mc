import pandas as pd
import numpy as np
import pickle
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_Selection import train_test_split
from sklearn.model_Selection import GridSearchCV
from sklearn.preprocessing import StandardScalar

df=pd.read_csv('120.csv')
df
#separating labels and features
Y = df['class'].values
Y = Y.astype('int')
X = df.drop(labels=['class'], axis=1)
#applied StandardScaler
sc = StandardScaler()
x = sc.fit_transform(X)
x
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42, shuffle=True)
#print (X_test)
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 90)
rf.fit (X_train, Y_train)
prediction_test = rf.predict(X_test)

from sklearn.model_selection import GridSearchCV
# Create the parameter grid based on the results of random search 
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 20, 30, 40, 50,60],
    'max_features': [2, 3],
    'min_samples_leaf': [1, 2, 3],
    'min_samples_split': [2, 4, 8],
    'n_estimators': [20, 40, 60, 80,100,120]
}
print(param_grid)

#Create a based model
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 5, n_jobs = -1, verbose = 2)

# Fit the grid search to the data
from sklearn.model_selection import GridSearchCV
grid_search.fit(X_train, Y_train)

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
RF_Model = RandomForestClassifier(bootstrap= 'True',
                                  max_depth = 30, 
                                  n_estimators = 40, max_features = 2, min_samples_leaf = 1, min_samples_split = 8, n_jobs =-1,)
RF_Model.fit(X_train,Y_train)

y_pred= RF_Model.predict(X_test)
y_pred
#dumping the model
import pickle
filename = 'My Final Model'
pickle.dump(RF_Model , open (filename , 'wb'))
#loading the model
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_test,Y_test)
return result
