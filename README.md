# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. Import the required libraries and create a dataset containing house area, number of bedrooms, price and occupants.
2. Separate the input features and target variables.
3. Train the multivariate linear regression model using SGDRegressor.
4. Predict the house price and number of occupants and display the results.

## Program:
```

Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: Jaden Samuel Abraham
RegisterNumber: 212225040138

import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

X=np.array([[1000,2],[1500,3],[2000,3],[2500,4],[3000,4],[3500,5]])
Y=np.array([[200000,3],[300000,4],[400000,5],[500000,5],[600000,6],[700000,7]])

scaler=StandardScaler()
X=scaler.fit_transform(X)

model=MultiOutputRegressor(SGDRegressor(max_iter=5000,learning_rate='constant',eta0=0.001,random_state=0))
model.fit(X,Y)

test=np.array([[2200,3]])
test=scaler.transform(test)

prediction=model.predict(test)

print("Predicted House Price:",prediction[0][0])
print("Predicted Number of Occupants:",prediction[0][1])

```

## Output:

<img width="1357" height="607" alt="image" src="https://github.com/user-attachments/assets/d0f2e415-7182-4a4f-b923-698a5966196f" />

## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
