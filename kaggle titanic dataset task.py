
import pandas as pd 
path = "/content/drive/MyDrive/Kaggle_Datasets/Titanic/train_and_test2.csv"
df = pd.read_csv(path)
df = df.drop(columns = [ 'zero', 'zero.1',
       'zero.2', 'zero.3', 'zero.4', 'zero.5', 'zero.6', 'zero.7',
       'zero.8', 'zero.9', 'zero.10', 'zero.11', 'zero.12', 'zero.13',
       'zero.14', 'zero.15', 'zero.16', 'zero.17',
       'zero.18'
])
df = df.rename(columns = {'2urvived':'Survived'})


import pandas as pd 
path = "/content/drive/MyDrive/Kaggle_Datasets/Titanic/train_and_test2.csv"
df = pd.read_csv(path)
df = df.drop(columns = [ 'zero', 'zero.1',
       'zero.2', 'zero.3', 'zero.4', 'zero.5', 'zero.6', 'zero.7',
       'zero.8', 'zero.9', 'zero.10', 'zero.11', 'zero.12', 'zero.13',
       'zero.14', 'zero.15', 'zero.16', 'zero.17',
       'zero.18'
])
df = df.rename(columns = {'2urvived':'Survived'})

import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import QuantileTransformer, PowerTransformer, OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split



num_features = ['Passengerid', 'Age', 'Fare', 'Sex', 'sibsp', 'Parch', 'Pclass',
       'Embarked', 'Survived'] 


num_imputer = KNNImputer(n_neighbors=5)

df[num_features] = num_imputer.fit_transform(df[num_features])


# Feature engineering
df['Family_Size'] = df['sibsp'] + df['Parch']
df['Is_Alone'] = (df['Family_Size'] == 0).astype(int)
df['Fare_Per_Person'] = df['Fare'] / (df['Family_Size'] + 1)

from sklearn.preprocessing import QuantileTransformer , PowerTransformer

scaler = ColumnTransformer([
    ('quantile',QuantileTransformer(output_distribution='normal'),['Fare', 'Fare_Per_Person']), 
    ('Power',PowerTransformer(), ['Age','Family_Size'])
],remainder = 'passthrough')


x = df.drop(columns=['Survived'])
y = df['Survived']
x_train, x_test, y_train, y_test  = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.ensemble import RandomForestClassifier 
from sklearn.metrics import accuracy_score
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
accuracy_no_preprocessing = accuracy_score(y_test, y_pred)

# Model with preprocessing
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)
clf.fit(x_train_scaled, y_train)
y_pred_scaled = clf.predict(x_test_scaled)
accuracy_with_preprocessing = accuracy_score(y_test, y_pred_scaled)

print(f'Accuracy without preprocessing: {accuracy_no_preprocessing:.4f}')
print(f'Accuracy with preprocessing: {accuracy_with_preprocessing:.4f}')
''' easily ''' 
