import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle


dataset = pd.read_csv("iris_dataset.csv",index_col = [0])

X = dataset.iloc[:, 0:4].values
y = dataset['Species'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=1000)

model.fit(X_train, y_train)

with open('iris_classifier.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)