import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier

data = pd.read_csv("tweets_2020-04-30.csv")

processed_data = []
for i in range(0, len(data)):
    processed_text = re.sub('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+','', str(data['text'][i])) #remove hyperlinks
    processed_text = re.sub("(@[A-Za-z0-9_]+)","", processed_text) #Remove @ mentions on tweets
    processed_text = re.sub(r'\W', ' ', processed_text) #Remove all special characters
    processed_text = re.sub(r'\s+[a-zA-Z]\s+', ' ', processed_text) #Remove single characters
    processed_text = re.sub(r'\^[a-zA-Z]\s+', ' ', processed_text)  #Remove single characters from the beginning of the sentence
    processed_text = re.sub(r'\s+', ' ', processed_text, flags=re.I) #Single space instead of multiple spaces
    processed_text = processed_text.lower() #Convert sentence to lowercase
    processed_data.append(processed_text)

#This part is to convert each tweet into a numerical representation using CountVectorizer
cv = CountVectorizer()
cv.fit(processed_data)
X = cv.transform(processed_data) #transform processed data into numeric representation
y = data["Mood"]
#Normalize the numerical representation of the tweets using preprocessing
data_normalized = preprocessing.normalize(X)

#Approach using a Stratified KFold Validation of 5 Folds
skf = StratifiedKFold(n_splits=5)


"""This part will train the model LinearSVC with different train and test indices
according to the number of splits obtained by StratifiedKFold"""

accuracies = []
for train_index, test_index in skf.split(data_normalized, y):
    X_train, X_test = data_normalized[train_index], data_normalized[test_index]
    y_train, y_test = y[train_index], y[test_index]
    classifier = LinearSVC()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    acc = classifier.score(X_test, y_test)
    accuracies.append(acc)
