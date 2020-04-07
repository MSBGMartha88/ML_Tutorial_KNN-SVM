# Manage Librarie requirements
# File>Settings>Projectname>ProjectInterpreter>Add(plus- upper right corner)specific libraries>Pandas>Install library

import pandas as pd

# attribute names
names = [ 'sepal_length', 'speal_width', 'petal_length', 'petal_width', 'class' ]

# get data with attribute names
iris_data = pd.read_csv ( 'iris_data.csv', names=names )

# shuffle data
iris_data = iris_data.sample ( frac=1 )

# print 5 first lines of the data
iris_data.head ()
print ( iris_data.head () )

# exctract feature variables
x_variables = iris_data.loc[:, iris_data.columns != 'class']

# extract target variable
y_variable = iris_data[ 'class' ]

from sklearn.model_selection import train_test_split

# get training and test data
x_train, x_test, y_train, y_test = train_test_split(x_variables, y_variable, test_size=0.20)

from sklearn.preprocessing import MinMaxScaler

#create MinMaxScaler object
scaler_min_max = MinMaxScaler()

#fit object to data
scaler_min_max.fit(x_train)

#get transformed train data
x_train_normalized = scaler_min_max.transform(x_train)

#get transformed test data
x_test_normalized = scaler_min_max.transform(x_test)

# Skalierte Daten ansehen #
print(x_train_normalized)
print(x_test_normalized)

# funktion=scaler_min_max
# die Werte über 1 und unter 0 wurden nun über eine Art Prozentsatz zu 0-100 oder 0-1 normalisiert

from sklearn.neighbors import KNeighborsClassifier

#create KNeighborsClassifier object
classifier_normalization = KNeighborsClassifier(n_neighbors=10)

#fit object to data
classifier_normalization.fit(x_train_normalized, y_train)

#Look at the algorythm parameters
print(KNeighborsClassifier)
print(classifier_normalization)

#get predicitons
y_pred_normalization = classifier_normalization.predict(x_test_normalized)

from sklearn.metrics import classification_report, confusion_matrix

#confusion matrix
print(confusion_matrix(y_test, y_pred_normalization))

#classifiaction report
print(classification_report(y_test, y_pred_normalization))

