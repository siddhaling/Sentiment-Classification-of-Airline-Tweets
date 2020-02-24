#Airline sentiment prediction using TF-IDF features. 
import re
import pandas as pd 
import numpy as np 
import warnings 
import os
from textblob import TextBlob
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore", category=DeprecationWarning)

#os.chdir('Specify current directory')
brand='easyjet'
porter=PorterStemmer()
#preprocess the tweets
def preprocess_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

def get_sentiment(tweet):
    ana=TextBlob(tweet)
    return(ana.sentiment)

data=pd.read_csv('easyjet.txt', sep=';' , header=None)
data.drop(data.index[0],axis=0,inplace=True)
data.drop(data.columns[2],axis=1,inplace=True)

data.columns=['date','tweet']
data['sentiment']=np.nan


data['cleanT']=data['tweet'].apply(preprocess_tweet)
data['sentiment']=data['cleanT'].map(get_sentiment)
data['tokens']=data['cleanT'].apply(lambda x:x.split())
data['stemmed']=data['tokens'].apply(lambda x: [porter.stem(i) for i in x])
data['cleanStemmedT']=data['stemmed'].apply(lambda x:' '.join(i for i in x))
data['polarity']=data['sentiment'].apply(lambda x: x.polarity)
data['sentiLabel']=data['polarity']>0


tfidf_vec=TfidfVectorizer(max_df=0.9,min_df=2, max_features=1000, stop_words='english')
tfidfFeatures=tfidf_vec.fit_transform(data['cleanStemmedT'])


xtrain_bag, xtest_bag, ytrain_bag, ytest_bag = train_test_split(tfidfFeatures, data['sentiLabel'], random_state=91, test_size=0.3)

logRec = LogisticRegression()
logRec.fit(xtrain_bag, ytrain_bag)
predictionTest = logRec.predict_proba(xtest_bag)
predBool = predictionTest[:,1] >= 0.3
predInt = predBool.astype(np.int)
f1_score(ytest_bag, predInt )
