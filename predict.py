# -*- coding: utf-8 -*-
"""
Created on Mon Feb 19 23:00:47 2018

@author: Piyushjaiswal
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print ("Select the Cryptocurrency to predict the opening price\n\n")
print ("1 : Bitcoin\n")
print ("2 : Ethereum\n")
print ("3 : Ripple\n")
print ("4 : Bitcoin Cash\n")
print ("5 : Cardano\n")
print ("6 : Stellar\n")
print ("7 : Litecoin\n")
print ("8 : NEO\n")
print ("9 : EOS\n")
print ("10 : NEM\n\n")
print ("Input your choice\n\n")

choice = int(input())
if (choice == 1):
    a = 0
    b = 1492
    c = "Bitcoin"
elif (choice == 2):
    a = 1493
    b = 2397
    c = "Ethereum"

elif (choice == 3):
    a = 2398
    b = 4035
    c = "Ripple"

elif (choice == 4):
    a = 4036
    b = 4224
    c = "Bitcoin Cash"

elif (choice == 5):
    a = 4225
    b = 4343
    c = "Cardano"

elif (choice == 6):
    a = 4344
    b = 5615
    c = "Stellar"

elif (choice == 7):
    a = 5616
    b = 7351
    c = "Litecoin"

elif (choice == 8):
    a = 7352
    b = 7810
    c = "NEO"

elif (choice == 9):
    a = 7811
    b = 8018
    c = "EOS"

elif (choice == 10):
    a = 8019
    b = 9051
    c = "NEM"

else:
    print("Invalid Choice\n")
    print("Please select a number between (1-10)\n")
    ch = int(input())
    if (ch == 1):
        a = 0
        b = 1492
        c = "Bitcoin"
    elif (ch == 2):
        a = 1493
        b = 2397
        c = "Ethereum"
    
    elif (ch == 3):
        a = 2398
        b = 4035
        c = "Ripple"
    
    elif (ch == 4):
        a = 4036
        b = 4224
        c = "Bitcoin Cash"
    
    elif (ch == 5):
        a = 4225
        b = 4343
        c = "Cardano"
    
    elif (ch == 6):
        a = 4344
        b = 5615
        c = "Stellar"
    
    elif (ch == 7):
        a = 5616
        b = 7351
        c = "Litecoin"
    
    elif (ch == 8):
        a = 7352
        b = 7810
        c = "NEO"
    
    elif (ch == 9):
        a = 7811
        b = 8018
        c = "EOS"
    
    elif (ch == 10):
        a = 8019
        b = 9051
        c = "NEM"
    

# Importing the dataset
dataset = pd.read_csv('ML_proj_dataset.csv')
X_mlr = dataset.iloc[a:b, [6,7,8,12]].values
X_svr = dataset.iloc[a:b, 6:7].values
X_close = dataset.iloc[a:b, 8:9].values
X_high = dataset.iloc[a:b, 6:7].values
X_low = dataset.iloc[a:b, 7:8].values
X_spread = dataset.iloc[a:b, 12:13].values
y = dataset.iloc[a+1:b+1, 5].values
y_svr = dataset.iloc[a+1:b+1, 5:6].values
y_svr = y_svr.reshape(-1,1)

print("\n")
print ("Input the parameters")

high = float(input())
low = float(input())
close = float(input())
spread = float(input())

arr = np.array([[high,low,close,spread]])
p_sum = 0
count = 0

# Feature Scaling for SVR
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_svr = sc_X.fit_transform(X_svr)
y_svr = sc_y.fit_transform(y_svr)

# Fitting the SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_svr, y_svr)

# Predicting a new result with SVR
y_pred4 = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[high]]))))

if ((y_pred4) >= low and (y_pred4) <=high):
    p_sum = p_sum + y_pred4
    count = count + 1

# Splitting the dataset into the Training set and Test set for Multiple Linear Regression
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_mlr, y, test_size = 0.2, random_state = 0)

#Fitting Multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the test set results using MLR
y_pred1 = regressor.predict(arr)

if ((y_pred1) >= low and (y_pred1) <=high):
    p_sum = p_sum + y_pred1
    count = count + 1
    
    
# Fitting the Random Forest Regression Model to the dataset and predicting the result

# High   
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
regressor.fit(X_high, y)

y_pred3_1  = regressor.predict(high)

if ((y_pred3_1) >= low and (y_pred3_1) <=high):
    p_sum = p_sum + y_pred3_1
    count = count + 1

# Close
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
regressor.fit(X_close, y)

y_pred3_2  = regressor.predict(close)

if ((y_pred3_2) >= low and (y_pred3_2) <=high):
    p_sum = p_sum + y_pred3_2
    count = count + 1

# Low
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
regressor.fit(X_low, y)

y_pred3_3  = regressor.predict(low)

if ((y_pred3_3) >= low and (y_pred3_3) <=high):
    p_sum = p_sum + y_pred3_3
    count = count + 1

# Spread
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 1000, random_state = 0)
regressor.fit(X_spread, y)

y_pred3_4  = regressor.predict(spread)

if ((y_pred3_4) >= low and (y_pred3_4) <=high):
    p_sum = p_sum + y_pred3_4
    count = count + 1
    
    
# Fitting Polynomial Regression to the dataset and predicting the result

print ("\n\nThe dependence of the close value against the various input parameters is shown as follows:\n\n")

# High  
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_high)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

y_pred2_1 = lin_reg2.predict(poly_reg.fit_transform(high))

if ((y_pred2_1) >= low and (y_pred2_1) <=high):
    p_sum = p_sum + y_pred2_1
    count = count + 1

plt.scatter(X_high, y, color = 'red')
plt.plot(X_high, regressor.predict(X_high), color = 'blue')
plt.title('Opening Price Vs High')
plt.xlabel('High')
plt.ylabel('Open')
plt.show()

# Close
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_close)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

y_pred2_2 = lin_reg2.predict(poly_reg.fit_transform(close))

if ((y_pred2_2) >= low and (y_pred2_2) <=high):
    p_sum = p_sum + y_pred2_2
    count = count + 1

plt.scatter(X_close, y, color = 'red')
plt.plot(X_close, regressor.predict(X_close), color = 'blue')
plt.title(' Opening Price Vs Closing Price ')
plt.xlabel('Close')
plt.ylabel('Open')
plt.show()

# Low
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_low)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

y_pred2_3 = lin_reg2.predict(poly_reg.fit_transform(low))

if ((y_pred2_3) >= low and (y_pred2_3) <=high):
    p_sum = p_sum + y_pred2_3
    count = count + 1

plt.scatter(X_low, y, color = 'red')
plt.plot(X_low, regressor.predict(X_low), color = 'blue')
plt.title('Opening Price Vs Low')
plt.xlabel('Low')
plt.ylabel('Open')
plt.show()

# Spread
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X_spread)
lin_reg2 = LinearRegression()
lin_reg2.fit(X_poly, y)

y_pred2_4 = lin_reg2.predict(poly_reg.fit_transform(spread))

if ((y_pred2_4) >= low and (y_pred2_4) <=high):
    p_sum = p_sum + y_pred2_4
    count = count + 1

plt.scatter(X_spread, y, color = 'red')
plt.plot(X_spread, regressor.predict(X_spread), color = 'blue')
plt.title('Opening Price Vs Spread')
plt.xlabel('Spread')
plt.ylabel('Open')
plt.show()

# Calculating the Final Predicted Value

prediction = p_sum/count

print ("\n\nThe Predicted Opening Price for " + c +"  for the following day is : " + str(prediction))




