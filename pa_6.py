#******************************Download sonar dataset. Check description of the data. Understand dataset, check for missing values if any. Perform dummy variable conversion if required. Design appropriate model to predict if the data point belongs to Rock or Mine class.******************************



import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
#data loading
datapath="/home/student/Desktop/rock1.csv"
data=pd.read_csv(datapath,sep=",",names=["V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25", "V26","V27","V28","V29","V30","V31","V32","V33","V34","V35","V36","V37","V38","V39","V40","V41","V42","V43","V44","V45", "V46", "V47","V48","V49","V50", "V51", "V52", "V53", "V54","V55", "V56", "V57", "V58", "V59", "V60", "Class"])


#Data Cleaning(Finding missing data)
print("Null present..?=",data.isnull().sum())

dic={"R":0,"M":1}
c=[]
for i in data["Class"]:
    c.append(dic[i])
data["Class_new"]=pd.Series(c)

#Splitting dataset into training and validation

X=data.drop(["Class","Class_new"],axis=1)
#print(X)
y=data["Class_new"]
#print(y)

x_train=X[:180]
x_test=X[180:]
y_train=y[:180]
y_test=y[180:]

model=RandomForestClassifier(n_estimators=700,max_features='log2')   
model.fit(x_train, y_train)
pred = model.predict(x_test)

label=["Rock","Mine"]
for i,j in zip(pd.Series(pred), y_test):
    print("predicted=",label[i],"\tActual=",label[j])

print('Accuracy Score=',metrics.accuracy_score(pred,y_test))


"""
--------------------------------------------output------------------------------------------------
student@student-OptiPlex-3020:~$ python pa_6.py
Null present..?= V1       0
V2       0
V3       0
V4       0
V5       0
V6       0
V7       0
V8       0
V9       0
V10      0
V11      0
V12      0
V13      0
V14      0
V15      0
V16      0
V17      0
V18      0
V19      0
V20      0
V21      0
V22      0
V23      0
V24      0
V25      0
V26      0
V27      0
V28      0
V29      0
V30      0
        ..
V32      0
V33      0
V34      0
V35      0
V36      0
V37      0
V38      0
V39      0
V40      0
V41      0
V42      0
V43      0
V44      0
V45      0
V46      0
V47      0
V48      0
V49      0
V50      0
V51      0
V52      0
V53      0
V54      0
V55      0
V56      0
V57      0
V58      0
V59      0
V60      0
Class    0
Length: 61, dtype: int64
predicted= Mine 	Actual= Mine
predicted= Mine 	Actual= Mine
predicted= Mine 	Actual= Mine
predicted= Mine 	Actual= Mine
predicted= Mine 	Actual= Mine
predicted= Mine 	Actual= Mine
predicted= Mine 	Actual= Mine
predicted= Mine 	Actual= Mine
predicted= Mine 	Actual= Mine
predicted= Mine 	Actual= Mine
predicted= Rock 	Actual= Mine
predicted= Rock 	Actual= Mine
predicted= Rock 	Actual= Mine
predicted= Rock 	Actual= Mine
predicted= Rock 	Actual= Mine
predicted= Mine 	Actual= Mine
predicted= Mine 	Actual= Mine
predicted= Mine 	Actual= Mine
predicted= Mine 	Actual= Mine
predicted= Mine 	Actual= Mine
predicted= Mine 	Actual= Mine
predicted= Mine 	Actual= Mine
predicted= Mine 	Actual= Mine
predicted= Mine 	Actual= Mine
predicted= Rock 	Actual= Mine
predicted= Rock 	Actual= Mine
predicted= Rock 	Actual= Mine
predicted= Rock 	Actual= Mine
Accuracy Score= 0.6785714285714286
student@student-OptiPlex-3020:~$ 
"""




