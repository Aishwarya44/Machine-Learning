#********************************************** Download auto mpg data from kaggle.https://www.kaggle.com/uciml/autompg-dataset . Design a model to predict mileage per gallon performance of a vehicle.*************************************************

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


datapath= "/home/student/Desktop/auto-mpg(1).csv"
data=pd.read_csv(datapath,sep=",")
#print(data)

#*********************************Data Cleaning*****************************************
data1 = data[data.horsepower != '?']


X=data1.drop(["mpg","car name"],axis=1)
#print(X)
y=data1["mpg"]

#print('y=',y)

#print(data1.isnull().any())

#*********************************split data*********************************************
x_train=X[:350]
x_test=X[350:]
y_train=y[:350]
y_test=y[350:]
#*********************************fitting and training data******************************
linear=linear_model.LinearRegression()
linear.fit(x_train,y_train)
predicted=linear.predict(x_test)

for i,j in zip(pd.Series(predicted),y_test):
    print('Predicted=',i,'\t Actual=',j)

print( 'mean_sq_err=',mean_squared_error(predicted,y_test))




#=====================================OutPut============================================================
"""
student@student-OptiPlex-3020:~$ python pa_8.py
Predicted= 33.560632551226774 	 Actual= 33.7
Predicted= 32.77634861744495 	 Actual= 32.4
Predicted= 30.876814141711446 	 Actual= 32.9
Predicted= 31.26140112825738 	 Actual= 31.6
Predicted= 26.469214016323686 	 Actual= 28.1
Predicted= 26.18909514811042 	 Actual= 30.7
Predicted= 28.741931363625884 	 Actual= 25.4
Predicted= 28.199163640702185 	 Actual= 24.2
Predicted= 23.857247814822134 	 Actual= 22.4
Predicted= 23.06360798373601 	 Actual= 26.6
Predicted= 25.947968737719545 	 Actual= 20.2
Predicted= 23.874173571664574 	 Actual= 17.6
Predicted= 29.039001616589797 	 Actual= 28.0
Predicted= 28.800780278228277 	 Actual= 27.0
Predicted= 30.2856709415157 	 Actual= 34.0
Predicted= 29.1879608033728 	 Actual= 31.0
Predicted= 29.844853060306722 	 Actual= 29.0
Predicted= 28.75016273212748 	 Actual= 27.0
Predicted= 27.722145716798234 	 Actual= 24.0
Predicted= 34.29443499114235 	 Actual= 36.0
Predicted= 35.393709503738705 	 Actual= 37.0
Predicted= 35.71586911966588 	 Actual= 31.0
Predicted= 32.14755744030842 	 Actual= 38.0
Predicted= 31.996694115016826 	 Actual= 36.0
Predicted= 34.59345045971408 	 Actual= 36.0
Predicted= 34.32991898916876 	 Actual= 36.0
Predicted= 34.236075115520734 	 Actual= 34.0
Predicted= 35.699674070896776 	 Actual= 38.0
Predicted= 35.71649789901129 	 Actual= 32.0
Predicted= 35.54492580635198 	 Actual= 38.0
Predicted= 26.745004389465908 	 Actual= 25.0
Predicted= 27.92090054415669 	 Actual= 38.0
Predicted= 29.626516427948363 	 Actual= 26.0
Predicted= 28.09987775432403 	 Actual= 22.0
Predicted= 31.717856645695438 	 Actual= 32.0
Predicted= 30.72129468897828 	 Actual= 36.0
Predicted= 27.41761692936848 	 Actual= 27.0
Predicted= 28.25606288771628 	 Actual= 27.0
Predicted= 33.8270334444055 	 Actual= 44.0
Predicted= 31.146619814601987 	 Actual= 32.0
Predicted= 29.1521007673318 	 Actual= 28.0
Predicted= 28.528093102567116 	 Actual= 31.0
mean_sq_err= 13.926463678743081
student@student-OptiPlex-3020:~$ 
"""
