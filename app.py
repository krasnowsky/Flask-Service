from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import re, nltk, stop_words
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, classification_report
from joblib import dump, load
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
import pickle

app = Flask(__name__)

def transformation(text):
        text = re.sub('[^A-Za-z]',' ',text)
        return ' '.join([nltk.stem.PorterStemmer().stem(word) for word in text.lower().split() if word not in stop_words.get_stop_words('english')])

@app.route("/check", methods=['POST'])
def check():
    with open('vectorizeFinal', 'rb') as handle:
        vectorize = pickle.load(handle)

    with open('modelFinal.sav', 'rb') as handle:
        bnb = pickle.load(handle)

    x = request.form['text']
    x = transformation(x)

    #print(clf.predict_log_proba(vectorize.transform([x])))

    #return jsonify
    output = bnb.predict(vectorize.transform([x]))

    #bnb.predict(x)
    #bnb.predict_log_proba(x)

    if output[0] == 0:
        return jsonify(response='Real')
    else:
        return jsonify(response='Fake')
        

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000)
