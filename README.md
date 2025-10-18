# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Load Data – Import the dataset containing employee details and their salaries.

2.Preprocess Data – Handle missing values, encode categorical variables, and split into training and test sets.

3.Initialize Model – Create a DecisionTreeRegressor with suitable parameters (e.g., max_depth=5).

4.Train Model – Fit the regressor using training data (model.fit(X_train, y_train)).

5.Predict & Evaluate – Predict salaries on test data and evaluate using metrics like MAE, MSE, and R² score.

6.Visualize & Interpret – Plot the tree and analyze feature importance for salary prediction.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: ABISHA RANI S
RegisterNumber:  212224040012
*/

import pandas as pd
data=pd.read_csv("/content/Salary.csv")
data.head()

data.info()
data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()
y=data["Salary"]
y.head()

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
y_pred

from sklearn import metrics
from sklearn.metrics import r2_score
r2 = r2_score (y_test, y_pred)
r2

dt.predict([[5,6]])
```

## Output:

Data head:

<img width="273" height="206" alt="image" src="https://github.com/user-attachments/assets/39cc58d6-cf67-4426-bb16-d4e5cfe0c536" />

Data info:

<img width="378" height="351" alt="image" src="https://github.com/user-attachments/assets/977a745d-d9df-47a1-b1ad-66b487245e76" />

Data head for salary:

<img width="240" height="207" alt="image" src="https://github.com/user-attachments/assets/1a0a3685-758f-45cf-88fb-7035b427ea80" />

Salary:

<img width="161" height="233" alt="image" src="https://github.com/user-attachments/assets/f0f71695-d285-454f-83a8-4d01b74c6b1a" />

Mean square error:

<img width="240" height="36" alt="image" src="https://github.com/user-attachments/assets/68e198e6-eec2-465a-84ba-e4566ab27a40" />

r2 value:

<img width="218" height="37" alt="image" src="https://github.com/user-attachments/assets/3730a413-5386-4693-9b52-f04ada7a17da" />

Prediction:

<img width="210" height="33" alt="image" src="https://github.com/user-attachments/assets/f2b31535-4826-4fdf-95d4-a59c6ba9a245" />


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
