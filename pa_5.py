#**********************Load dow jones sensex data. Use linear regression to predict open, close and low prices.(3 different regression models to predict each open, close and low price one at a time.)******************************



import pandas as pd
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn import linear_model

#Loading data
dataPath="/home/student/dow_jones_index/dow_jones_index_data.csv"
data=pd.read_csv(dataPath,sep=",")

#print data
#print len(data["stock"].unique())

#data Preprocessing
stock_dummy=[]
st={
    'AA':0, 'AXP':1, 'BA':2, 'BAC':3, 'CAT':4, 'CSCO':5, 'CVX':6, 'DD':7, 'DIS':8, 'GE':9, 'HD':10, 'HPQ':11, 'IBM':12,
    'INTC':13, 'JNJ':14, 'JPM':15, 'KRFT':16, 'KO':17, 'MCD':18, 'MMM':19, 'MRK':20, 'MSFT':21, 'PFE':22, 'PG':23, 'T':24,
    'TRV':25, 'UTX':26, 'VZ':27, 'WMT':28, 'XOM':29
   }

for s in data["stock"]:
    stock_dummy.append(st[s])

data["stock_dummy"]=pd.Series(stock_dummy)
			
next_open=[]
next_close=[]
o=[]
h=[]
lo=[]
c=[]
for i,j,k,l,m,n in zip(data["next_weeks_open"],data["next_weeks_close"],data["open"],data["high"],data["low"],data["close"]):
    i=float(i.replace("$","")) #replacing $ with empty space because with $ it will consider the value as string
    j=float(j.replace("$",""))
    k=float(k.replace("$",""))
    l=float(l.replace("$",""))
    m=float(m.replace("$",""))
    n=float(n.replace("$",""))
    next_open.append(i)
    next_close.append(j)
    o.append(k)
    h.append(l)
    lo.append(m)
    c.append(n)
data["next_weeks_open_new"]=pd.Series(next_open)
data["next_weeks_close_new"]=pd.Series( next_close)
data["open_new"]=pd.Series(o)
data["high_new"]=pd.Series(h)
data["low_new"]=pd.Series(lo)
data["close_new"]=pd.Series(c)

data["date"]=pd.to_datetime(data["date"],format="%m/%d/%Y")

data.fillna(data.mean(),inplace=True)#This is done to replace "na" with mean
#X=data.drop(["stock","next_weeks_open","next_weeks_close","open","high","low","close","percent_return_next_dividend"],axis=1)


print("--------------------------------------To predict open value-------------------------------------------------")
X=data.drop(["next_weeks_open","next_weeks_close","open","open_new","high","close","low","date","stock"],axis=1)
y=data["open_new"]


#Spliting data in training and testing
x_train=X[:730]
x_test=X[730:]
y_train=y[:730]
y_test=y[730:]

#create linear regression object
linear=linear_model.LinearRegression()

#Train model
linear.fit(x_train,y_train)

#predict Output
predicted=linear.predict(x_test)

for i,j in zip(pd.Series(predicted),y_test):
    print("Predicted=",round(i,1),'\t',"Actual=",round(j,1))
print('mean_sq_err=',mean_squared_error(predicted,y_test))


print("--------------------------------------To predict close value----------------------------------")
X=data.drop(["next_weeks_open","next_weeks_close","open","close_new","high","close","low","date","stock"],axis=1)
y=data["close_new"]


#Spliting data in training and testing
x_train=X[:730]
x_test=X[730:]
y_train=y[:730]
y_test=y[730:]


#create linear regression object
linear=linear_model.LinearRegression()

#Train model
linear.fit(x_train,y_train)

#predict Output
predicted=linear.predict(x_test)


for i,j in zip(pd.Series(predicted),y_test):
    print("Predicted=",round(i,1),'\t',"Actual=",round(j,1))
print('mean_sq_err=',mean_squared_error(predicted,y_test))





print("--------------------------------------To predict low price-------------------------------------------------")
X=data.drop(["next_weeks_open","next_weeks_close","open","open_new","high","close","low","date","stock","low_new"],axis=1)
y=data["low_new"]



#Spliting data in training and testing
x_train=X[:730]
x_test=X[730:]
y_train=y[:730]
y_test=y[730:]

#create linear regression object
linear=linear_model.LinearRegression()

#Train model
linear.fit(x_train,y_train)

#predict Output
predicted=linear.predict(x_test)

for i,j in zip(pd.Series(predicted),y_test):
    print("Predicted=",round(i,1),'\t',"Actual=",round(j,1))

print('\n\nmean_sq_err=',mean_squared_error(predicted,y_test))



"""
#-------------------------------------------Output:----------------------------------------------

student@student-OptiPlex-3020:~$ python pa_5.py
--------------------------------------To predict open value-------------------------------------------------
Predicted= 55.0 	 Actual= 55.0
Predicted= 55.7 	 Actual= 55.6
Predicted= 55.0 	 Actual= 54.9
Predicted= 54.8 	 Actual= 54.9
Predicted= 53.9 	 Actual= 53.9
Predicted= 52.6 	 Actual= 52.9
Predicted= 53.0 	 Actual= 52.7
Predicted= 83.5 	 Actual= 83.9
Predicted= 84.6 	 Actual= 84.3
Predicted= 84.8 	 Actual= 86.0
Predicted= 83.7 	 Actual= 83.1
Predicted= 86.5 	 Actual= 86.3
Predicted= 86.9 	 Actual= 88.1
Predicted= 82.4 	 Actual= 83.0
Predicted= 80.7 	 Actual= 80.2
Predicted= 80.8 	 Actual= 80.2
Predicted= 82.7 	 Actual= 83.3
Predicted= 81.0 	 Actual= 80.9
Predicted= 79.9 	 Actual= 80.0
Predicted= 79.3 	 Actual= 78.7
mean_sq_err= 0.26804629317898054
student@student-OptiPlex-3020:~$ 
student@student-OptiPlex-3020:~$ 
student@student-OptiPlex-3020:~$ 
student@student-OptiPlex-3020:~$ python pa_5.py
--------------------------------------To predict open value-------------------------------------------------
Predicted= 55.0 	 Actual= 55.0
Predicted= 55.7 	 Actual= 55.6
Predicted= 55.0 	 Actual= 54.9
Predicted= 54.8 	 Actual= 54.9
Predicted= 53.9 	 Actual= 53.9
Predicted= 52.6 	 Actual= 52.9
Predicted= 53.0 	 Actual= 52.7
Predicted= 83.5 	 Actual= 83.9
Predicted= 84.6 	 Actual= 84.3
Predicted= 84.8 	 Actual= 86.0
Predicted= 83.7 	 Actual= 83.1
Predicted= 86.5 	 Actual= 86.3
Predicted= 86.9 	 Actual= 88.1
Predicted= 82.4 	 Actual= 83.0
Predicted= 80.7 	 Actual= 80.2
Predicted= 80.8 	 Actual= 80.2
Predicted= 82.7 	 Actual= 83.3
Predicted= 81.0 	 Actual= 80.9
Predicted= 79.9 	 Actual= 80.0
Predicted= 79.3 	 Actual= 78.7
mean_sq_err= 0.26804629317898054
--------------------------------------To predict close value----------------------------------
Predicted= 55.7 	 Actual= 55.7
Predicted= 55.2 	 Actual= 55.3
Predicted= 54.9 	 Actual= 54.7
Predicted= 53.8 	 Actual= 53.7
Predicted= 53.0 	 Actual= 52.7
Predicted= 52.7 	 Actual= 52.8
Predicted= 52.6 	 Actual= 52.4
Predicted= 84.3 	 Actual= 84.7
Predicted= 85.9 	 Actual= 86.0
Predicted= 83.6 	 Actual= 84.3
Predicted= 86.0 	 Actual= 86.4
Predicted= 87.8 	 Actual= 88.0
Predicted= 83.3 	 Actual= 82.7
Predicted= 80.7 	 Actual= 80.9
Predicted= 80.8 	 Actual= 81.6
Predicted= 82.9 	 Actual= 82.6
Predicted= 81.2 	 Actual= 81.2
Predicted= 80.2 	 Actual= 79.8
Predicted= 78.9 	 Actual= 79.0
Predicted= 77.5 	 Actual= 76.8
mean_sq_err= 0.14849483657963014
--------------------------------------To predict low price-------------------------------------------------
Predicted= 54.7 	 Actual= 54.7
Predicted= 54.5 	 Actual= 55.0
Predicted= 54.1 	 Actual= 54.1
Predicted= 53.2 	 Actual= 53.0
Predicted= 52.3 	 Actual= 52.7
Predicted= 52.0 	 Actual= 51.8
Predicted= 51.8 	 Actual= 52.4
Predicted= 83.1 	 Actual= 82.6
Predicted= 84.1 	 Actual= 84.1
Predicted= 83.4 	 Actual= 82.4
Predicted= 84.3 	 Actual= 82.4
Predicted= 85.8 	 Actual= 85.9
Predicted= 82.8 	 Actual= 81.6
Predicted= 80.4 	 Actual= 79.4
Predicted= 80.2 	 Actual= 79.6
Predicted= 80.7 	 Actual= 80.1
Predicted= 80.5 	 Actual= 80.2
Predicted= 78.8 	 Actual= 79.7
Predicted= 78.1 	 Actual= 78.3
Predicted= 76.2 	 Actual= 76.8


mean_sq_err= 0.4941643012806531


"""
