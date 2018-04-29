#*********************** Download titanic dataset from Kaggle. Design a model to find age of passenger for which it is missing. For this first separate rows with age columns present from entire data. Treat it as training data. Split training data into train and validation dataset. Use appropriate  algorithm to create model. Now separate rows with missing age column and treat it as test dataset. Predict age values on test dataset.***********************************


import pandas as pd
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

data=pd.read_csv("/home/student/Desktop/titanic/train.csv")

df= data[data['Age'].isnull()]
#print('df=',df)

#data Preprocessing
dic={"male":0,"female":1}
g=[]
for i in data["Sex"]:
    g.append(dic[i])
data["gender"]=pd.Series(g)

data.dropna(subset=['Age'], inplace = True)


#splitting data
x_train=data.drop(["Sex","Name","Cabin","Embarked","Ticket","gender","Age"],axis=1)
print(x_train.isnull().sum())
#print(X)

y_train=data["Age"]
#print("y_train=",y_train)

x_test=df.drop(["Sex","Name","Cabin","Embarked","Ticket","Age"],axis=1)
#print(x_test)

y_test=df["Age"]
print(y_test)

#Fitting and training data

linear=linear_model.LinearRegression()
linear.fit(x_train,y_train)

#predicting
predicted=linear.predict(x_test)

for i,j in zip(pd.Series(predicted),y_test):
    print('Predicted=',round(i,3),'\t Actual=',j)





"""
----------------------------------------------Output--------------------------------------------------
student@student-OptiPlex-3020:~$ python pa_2.py
PassengerId    0
Survived       0
Pclass         0
SibSp          0
Parch          0
Fare           0
dtype: int64
5     NaN
17    NaN
19    NaN
26    NaN
28    NaN
29    NaN
31    NaN
32    NaN
36    NaN
42    NaN
45    NaN
46    NaN
47    NaN
48    NaN
55    NaN
64    NaN
65    NaN
76    NaN
77    NaN
82    NaN
87    NaN
95    NaN
101   NaN
107   NaN
109   NaN
121   NaN
126   NaN
128   NaN
140   NaN
154   NaN
       ..
718   NaN
727   NaN
732   NaN
738   NaN
739   NaN
740   NaN
760   NaN
766   NaN
768   NaN
773   NaN
776   NaN
778   NaN
783   NaN
790   NaN
792   NaN
793   NaN
815   NaN
825   NaN
826   NaN
828   NaN
832   NaN
837   NaN
839   NaN
846   NaN
849   NaN
859   NaN
863   NaN
868   NaN
878   NaN
888   NaN
Name: Age, Length: 177, dtype: float64
Predicted= 28.951 	 Actual= nan
Predicted= 30.131 	 Actual= nan
Predicted= 22.189 	 Actual= nan
Predicted= 28.976 	 Actual= nan
Predicted= 22.183 	 Actual= nan
Predicted= 28.967 	 Actual= nan
Predicted= 32.188 	 Actual= nan
Predicted= 22.186 	 Actual= nan
Predicted= 22.195 	 Actual= nan
Predicted= 28.972 	 Actual= nan
Predicted= 28.971 	 Actual= nan
Predicted= 24.874 	 Actual= nan
Predicted= 22.191 	 Actual= nan
Predicted= 20.796 	 Actual= nan
Predicted= 37.838 	 Actual= nan
Predicted= 44.742 	 Actual= nan
Predicted= 17.305 	 Actual= nan
Predicted= 28.984 	 Actual= nan
Predicted= 28.982 	 Actual= nan
Predicted= 22.203 	 Actual= nan
Predicted= 28.985 	 Actual= nan
Predicted= 28.988 	 Actual= nan
Predicted= 28.992 	 Actual= nan
Predicted= 22.211 	 Actual= nan
Predicted= 17.982 	 Actual= nan
Predicted= 28.997 	 Actual= nan
Predicted= 29.003 	 Actual= nan
Predicted= 17.221 	 Actual= nan
Predicted= 27.308 	 Actual= nan
Predicted= 29.019 	 Actual= nan
Predicted= 29.0 	 Actual= nan
Predicted= -5.386 	 Actual= nan
Predicted= 36.791 	 Actual= nan
Predicted= 44.805 	 Actual= nan
Predicted= 16.003 	 Actual= nan
Predicted= -5.379 	 Actual= nan
Predicted= 36.942 	 Actual= nan
Predicted= 44.451 	 Actual= nan
Predicted= 18.137 	 Actual= nan
Predicted= 29.027 	 Actual= nan
Predicted= 22.243 	 Actual= nan
Predicted= -5.371 	 Actual= nan
Predicted= 25.047 	 Actual= nan
Predicted= 29.034 	 Actual= nan
Predicted= 16.021 	 Actual= nan
Predicted= 29.044 	 Actual= nan
Predicted= 24.956 	 Actual= nan
Predicted= 18.156 	 Actual= nan
Predicted= 29.053 	 Actual= nan
Predicted= 37.256 	 Actual= nan
Predicted= 29.049 	 Actual= nan
Predicted= 29.051 	 Actual= nan
Predicted= 44.764 	 Actual= nan
Predicted= 22.269 	 Actual= nan
Predicted= 37.2 	 Actual= nan
Predicted= 44.843 	 Actual= nan
Predicted= 44.822 	 Actual= nan
Predicted= 37.997 	 Actual= nan
Predicted= 22.278 	 Actual= nan
Predicted= 14.075 	 Actual= nan
Predicted= 30.24 	 Actual= nan
Predicted= 29.06 	 Actual= nan
Predicted= 36.8 	 Actual= nan
Predicted= -5.329 	 Actual= nan
Predicted= 14.085 	 Actual= nan
Predicted= 32.484 	 Actual= nan
Predicted= 29.073 	 Actual= nan
Predicted= 18.184 	 Actual= nan
Predicted= 44.732 	 Actual= nan
Predicted= 29.09 	 Actual= nan
Predicted= 22.296 	 Actual= nan
Predicted= 22.297 	 Actual= nan
Predicted= 24.983 	 Actual= nan
Predicted= 22.309 	 Actual= nan
Predicted= 22.302 	 Actual= nan
Predicted= 33.266 	 Actual= nan
Predicted= 29.09 	 Actual= nan
Predicted= 29.094 	 Actual= nan
Predicted= 16.083 	 Actual= nan
Predicted= 29.099 	 Actual= nan
Predicted= 29.115 	 Actual= nan
Predicted= 37.247 	 Actual= nan
Predicted= 29.098 	 Actual= nan
Predicted= 29.102 	 Actual= nan
Predicted= 29.114 	 Actual= nan
Predicted= 29.107 	 Actual= nan
Predicted= 18.213 	 Actual= nan
Predicted= 22.323 	 Actual= nan
Predicted= 24.947 	 Actual= nan
Predicted= 29.112 	 Actual= nan
Predicted= 33.747 	 Actual= nan
Predicted= 29.118 	 Actual= nan
Predicted= 29.115 	 Actual= nan
Predicted= 37.265 	 Actual= nan
Predicted= 29.121 	 Actual= nan
Predicted= 29.129 	 Actual= nan
Predicted= 44.522 	 Actual= nan
Predicted= 37.27 	 Actual= nan
Predicted= 16.11 	 Actual= nan
Predicted= 24.96 	 Actual= nan
Predicted= 29.03 	 Actual= nan
Predicted= 29.021 	 Actual= nan
Predicted= 29.135 	 Actual= nan
Predicted= 38.128 	 Actual= nan
Predicted= 29.131 	 Actual= nan
Predicted= 28.893 	 Actual= nan
Predicted= 29.148 	 Actual= nan
Predicted= 29.148 	 Actual= nan
Predicted= 42.008 	 Actual= nan
Predicted= 29.151 	 Actual= nan
Predicted= 20.552 	 Actual= nan
Predicted= 29.045 	 Actual= nan
Predicted= 30.302 	 Actual= nan
Predicted= 29.149 	 Actual= nan
Predicted= 41.932 	 Actual= nan
Predicted= 29.153 	 Actual= nan
Predicted= 29.149 	 Actual= nan
Predicted= 29.15 	 Actual= nan
Predicted= 29.163 	 Actual= nan
Predicted= 22.373 	 Actual= nan
Predicted= 25.073 	 Actual= nan
Predicted= 29.147 	 Actual= nan
Predicted= 29.158 	 Actual= nan
Predicted= 27.576 	 Actual= nan
Predicted= 30.033 	 Actual= nan
Predicted= 29.174 	 Actual= nan
Predicted= 29.165 	 Actual= nan
Predicted= 44.709 	 Actual= nan
Predicted= 29.181 	 Actual= nan
Predicted= 18.284 	 Actual= nan
Predicted= 29.171 	 Actual= nan
Predicted= 29.177 	 Actual= nan
Predicted= 45.352 	 Actual= nan
Predicted= 25.069 	 Actual= nan
Predicted= 21.67 	 Actual= nan
Predicted= 29.186 	 Actual= nan
Predicted= 29.182 	 Actual= nan
Predicted= 22.399 	 Actual= nan
Predicted= 29.184 	 Actual= nan
Predicted= 29.189 	 Actual= nan
Predicted= 33.818 	 Actual= nan
Predicted= 37.337 	 Actual= nan
Predicted= 29.189 	 Actual= nan
Predicted= 21.687 	 Actual= nan
Predicted= 22.416 	 Actual= nan
Predicted= 17.527 	 Actual= nan
Predicted= 44.983 	 Actual= nan
Predicted= 29.092 	 Actual= nan
Predicted= 22.426 	 Actual= nan
Predicted= 37.357 	 Actual= nan
Predicted= 29.212 	 Actual= nan
Predicted= 29.212 	 Actual= nan
Predicted= 38.157 	 Actual= nan
Predicted= 29.121 	 Actual= nan
Predicted= 44.807 	 Actual= nan
Predicted= 24.994 	 Actual= nan
Predicted= 29.234 	 Actual= nan
Predicted= 29.227 	 Actual= nan
Predicted= 29.228 	 Actual= nan
Predicted= 23.421 	 Actual= nan
Predicted= 29.232 	 Actual= nan
Predicted= -5.167 	 Actual= nan
Predicted= 44.949 	 Actual= nan
Predicted= 45.415 	 Actual= nan
Predicted= 29.256 	 Actual= nan
Predicted= 28.518 	 Actual= nan
Predicted= 22.461 	 Actual= nan
Predicted= 29.255 	 Actual= nan
Predicted= 29.244 	 Actual= nan
Predicted= 38.195 	 Actual= nan
Predicted= -5.149 	 Actual= nan
Predicted= 33.327 	 Actual= nan
Predicted= 29.264 	 Actual= nan
Predicted= -5.143 	 Actual= nan
Predicted= 29.233 	 Actual= nan
Predicted= 29.26 	 Actual= nan
Predicted= 23.457 	 Actual= nan
student@student-OptiPlex-3020:~$ 
"""



