import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

class Predictor:
    def __init__(self, datafile, classifier):
        self.data = pd.read_csv(datafile)
        self.classifier = classifier #LinearSVC() or KNeigborsClassifier()
    
    def clean_data(self):
        self.processed_data = []
        for i in range(0, len(self.data)):
            processed_text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', str(self.data['text'][i]))
            processed_text = re.sub("(@[A-Za-z0-9_]+)","", processed_text)
            processed_text = re.sub(r'\W', ' ', processed_text)
            processed_text = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_text)
            processed_text = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_text) 
            processed_text = re.sub(r'\s+', ' ', processed_text, flags=re.I)
            processed_text = processed_text.lower()
            self.processed_data.append(processed_text)
        return self.processed_data
    
    def transform_normalize(self, processed_data):
        cv = CountVectorizer()
        cv.fit(processed_data)
        X = cv.transform(processed_data) #transformed data into numeric representation
        self.data_normalized = preprocessing.normalize(X)
        return self.data_normalized

    def split(self, data_normalized, test_size):
        X = data_normalized
        y = np.asarray(self.data["Mood"])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size = test_size, random_state = 3)

    def fit(self):
        self.predictor = self.classifier.fit(self.X_train, self.y_train)
    
    def train_model(self, data_normalized, test_size):
        self.split(data_normalized, test_size)
        self.fit()
        return self.predictor


    def predict(self, input_value):
        if input_value == None:
            result = self.classifier.predict(self.X_test)
        else:
            result = self.classifier.predict(np.array([input_values]))
        return result

if __name__ == '__main__':
    predictor_model = Predictor("tweets_2020-04-30.csv", KNeighborsClassifier())
    predictor_model.clean_data()
    predictor_model.transform_normalize(predictor_model.processed_data)
    predictor_model.split(predictor_model.data_normalized,0.2)
    predictor_model.fit()
    #You can use predictor_model.train_model(predictor_model.data_normalized,0.2) instead of split() and fit()
    print("Accuracy: ", predictor_model.predictor.score(predictor_model.X_test, predictor_model.y_test))