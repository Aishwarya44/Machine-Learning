#******************Collect tweets from twitter api.i.e. download twitter data. Perform text/string analysis on data to find if its sentiment is positive or negative.*********************************

import nltk, os
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer
from sklearn.naive_bayes import GaussianNB
import pickle

class EmotionClassifier():
    dataPath= "/home/student/emotions/"

    def __init__(self):#__init__=default method
        self.labels=['Happy','anxiety','Anger','peace','sad']
        self.vectorizer=TfidfVectorizer(max_features=None,strip_accents='unicode',analyzer='words',ngram_range=(1,3),use_idf=1,smooth_idf = 1,stop_words = 'english')

    def getData(self):
        
        x_data=[]
        y_data=[]
        labels=['Happy','anxiety','Anger','peace','sad']
        for label in labels:
            with open(self.dataPath + label + ".txt") as file_name:
                data_file=file_name.readlines()
                for data in data_file:
                    tokens=nltk.word_tokenize(data)
                    stems=[]
                    for item in map(lambda x : x.decode("utf-8"), tokens):
                        stems.append(PorterStemmer().stem(item))
                    data = " ".join(stems)
                    x_data.append(data)
                    y_data.append(self.labels.index(label))
        return x_data,y_data 

    def train(self):
        C=0.01
        x_train, y_train = self.getData()
        vectors_train = self.vectorizer.fit_transform(x_train)
        if os.path.exists("./emotion_classifier.pkl"):
            with open('./emotion_classifier.pkl', 'rb') as fid:
                self.classifier = pickle.load(fid)
                self.classifier.fit(vectors_train, y_train)
        else:
            self.classifier = GaussianNB()
            self.classifier.fit(vectors_train, y_train)
            with open('./emotion_classifier.pkl', 'wb+') as fid:
                pickle.dump(self.classifier, fid)

    def test(self, query):
        self.train()
        tokens = nltk.word_tokenize(query)
        stems = []
        for item in map(lambda x : x.decode("utf-8"), tokens):
            stems.append(PorterStemmer().stem(item))
        query = " ".join(stems)
        classifier_result = {"class":"", "confidence":""}
        vectors_test = self.vectorizer.transform([query])
        pred = self.classifier.predict(vectors_test)
        predp = self.classifier.predict_proba(vectors_test)
        predp = predp.tolist()[0]
        print("predp : : ", predp)
        max_ind =  predp.index(max(predp))
        classifier_result["class"] = self.labels[max_ind]
        classifier_result["confidence"] = str(max(predp)*100.0)
        
        return pred,self.labels[pred[0]]


if __name__ == '__main__':
    log = EmotionClassifier()
    while(True):
        query = input("=========Enter Query============\n\n")
        result = log.test(query)
        print(result)
        print("\n")







