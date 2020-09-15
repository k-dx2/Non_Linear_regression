import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

def sigmoid(x, Beta_1, Beta_2):
     y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
     return y

#importing the dataset
df = pd.read_csv("china_gdp.csv")
print(df.head(10))

#plotting the data
x_data = df["Year"].values
y_data = df["Value"].values
plt.figure(figsize=(8,5))
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()


# Lets normalize our data
x_data =x_data/max(x_data)
y_data =y_data/max(y_data)

#splitting the dataset into test and train
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=3)


popt, pcov = curve_fit(sigmoid, X_train, y_train)
#print the final parameters
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

#plotting the Regression model
x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8,5))
y = sigmoid(x, *popt)
plt.plot(x_data, y_data, 'ro', label='data')
plt.plot(x,y, linewidth=3.0, label='fit')
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

#testing metrics
test_y_hat = sigmoid(X_test,*popt)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_hat - y_test)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_hat - y_test) ** 2))
print("R2-score: %.2f" % r2_score(test_y_hat , y_test) )




