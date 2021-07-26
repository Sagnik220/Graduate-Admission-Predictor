import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, accuracy_score


data=pd.read_csv('Admission_Predict_Ver1.1.csv',index_col='Serial No.')

y=data['Chance of Admit ']
X=data.drop(['Chance of Admit '],axis=1)

X=np.array(X)
y=np.array(y)
y=y.reshape(-1,1)

# scaling the data before training the model
scaler_x = StandardScaler()
X = scaler_x.fit_transform(X)
scaler_y = StandardScaler()
y = scaler_y.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,shuffle=True)

linear_regression_model = LinearRegression()
linear_regression_model.fit(X_train, y_train)

model=joblib.dump(linear_regression_model,'linear_reg_model.pkl')
