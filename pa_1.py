#****************************************Load bank.csv data into python script. Split data into validation and training dataset. Design decision tree algorithm to predict if individual can get loan or not.******************************************************


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics

#loading dataset
datapath="/home/student/bank/bank.csv"
data=pd.read_csv(datapath,sep=";")
#print(data)

#preprocessing

#preprocessing poutcome
a={"unknown":0,"other":1,"failure":2,"success":3}
pout=[]
for i in data["poutcome"]:
    pout.append(a[i])
data["poutcome_new"]=pd.Series(pout)
#print(data)

#preprocessing marital
b={ "married":0,"divorced":1,"single":2}
mar=[]
for i in data["marital"]:
    mar.append(b[i])
data["marital_new"]=pd.Series(mar)

#preprocessing education
e={"unknown":0,"secondary":1,"primary":2,"tertiary":3}
edu=[]
for i in data["education"]:
    edu.append(e[i])
data["education_new"]=pd.Series(edu)

#preprocessing job
j={"admin.":0,"unknown":1,"unemployed":2,"management":3,"housemaid":4,"entrepreneur":5,"student":6,"blue-collar":7,"self-employed":8,"retired":9,"technician":10,"services":11}
jo=[]
for i in data["job"]:
    jo.append(j[i])
data["job_new"]=pd.Series(jo)

#preprocessing default,loan and y
dic={"yes":1,"no":0}
d=[]
l=[]
y1=[]
ho=[]
for i,j,k,h in zip(data["default"],data["loan"],data["y"],data["housing"]):
    d.append(dic[i])
    l.append(dic[j])
    y1.append(dic[k])
    ho.append(dic[h])
data["default_new"]=pd.Series(d)
data["loan_new"]=pd.Series(l)
data["y_new"]=pd.Series(y1)
data["housing_new"]=pd.Series(ho)

#preprocessing month
month={"jan":1,"feb":2,"mar":3,"apr":4,"may":5,"jun":6,"jul":7,"aug":8,"sep":9,"oct":10,"nov":11,"dec":12}
mo=[]
for m in data["month"]:
    mo.append(month[m])
data["month_new"]=pd.Series(mo)

#splitting dataset into training and validation
X=data.drop(["poutcome","marital","education","job","default","loan","y","housing","month","y_new","contact","month_new","day","pdays"],axis=1)
y=data["y_new"]
#print(y)

#splitting data
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=123,stratify=y)
print("x_train=",len(x_train))
print("x_test=",len(x_test))
print("y_train=",len(y_train))
print("y_test=",len(y_test))

#fitting to model
model = tree.DecisionTreeClassifier(criterion='gini')
model.fit(x_train,y_train)
pred=model.predict(x_test)

label=["no","yes"]
#printing output
for i,j in zip(pd.Series(pred),y_test):
    print("Predicted=",label[i],'\t',"Actual=",label[j])

print('Accuracy Score=', metrics.accuracy_score(pred,y_test))


#*************************************************         OUTPUT          *************************************************
'''
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= yes 	 Actual= yes
Predicted= yes 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= yes 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= yes 	 Actual= yes
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= yes 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= yes
Predicted= no 	 Actual= no
Predicted= yes 	 Actual= yes
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= yes 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= yes
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= yes
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= yes 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= yes
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= yes 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= yes
Predicted= yes 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= no 	 Actual= no
Predicted= yes 	 Actual= no
Predicted= yes 	 Actual= yes
Accuracy Score= 0.8596685082872928
'''

