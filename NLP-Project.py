#!/usr/bin/env python
# coding: utf-8

# In[2]:


#STOPWORDS REMOVAL

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
class TextPreprocessor:
    def remove_stopwords(self,text):
        self.stop_words = set(stopwords.words("english"))
        self.words = word_tokenize(text)
        filtered_text = [w for w in self.words if not w in self.stop_words]
        return filtered_text


if __name__=="__main__":
    nltk.download('stopwords')
    nltk.download("punkt")
    text = "This is a sample text, showing off the stop words filtration."
    tp = TextPreprocessor()
    print(tp.remove_stopwords(text))


# In[4]:


#STEMMING

# STEMMING IS THE PROCESS OF REDUCING INFLECTION IN WORDS
# TO THEIR ROOT FORMS SUCH AS MAPPING A GROUP OF WORDS TO THE SAME 
# STEM EVEN IF THE STEM IS NOT A VALID WORD IN THE LANGUAGE

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import nltk
stem=PorterStemmer()

def s_words(text):
    words=word_tokenize(text)
    stems=[stem.stem(word) for word in words]
    return stems

if __name__=="__main__":
    text = "This is a sample text, showing off the stop words filtration."
    print(s_words(text))


# In[7]:


#lemmatization
#  LEMMATIZATION IS THE PROCESS OF GROUPING TOGETHER THE

# INFLECTED FORMS OF A WORD SO THEY CAN BE ANALYZED AS A SINGLE ITEM

#it will help us to find the similarity between two words
#more menaingful words
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import nltk
lemmatizer=WordNetLemmatizer()

def l_words(text):
    words=word_tokenize(text)
    lemmas=[lemmatizer.lemmatize(word) for word in words]
    return lemmas

if __name__=="__main__":
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    text = "This is a sample text, showing off the stop words filtration."
    print(l_words(text))


# In[13]:


#POS TAGGING
# PARTS OF SPEECH TAGGING IS THE PROCESS OF CLASSIFYING WORDS INTO THEIR
# RESPECTIVE PARTS OF SPEECH AND LABELING THEM ACCORDINGLY

from nltk import pos_tag
from nltk.tokenize import word_tokenize
import nltk
def pos_tagging(text):
    words=word_tokenize(text)
    pos=pos_tag(words)
    return pos

if __name__=="__main__":
    nltk.download('averaged_perceptron_tagger')
    text = "This is a sample text, showing off the stop words filtration."
    print(pos_tagging(text))


# In[17]:


#named entity recognition
# NAMED ENTITY RECOGNITION IS THE PROCESS OF IDENTIFYING AND CLASSIFYING

# ENTITIES IN TEXT INTO PREDEFINED CATEGORIES SUCH AS THE NAMES OF PERSONS, ORGANIZATIONS, LOCATIONS, MONETARY VALUES, PERCENTAGES, TIMES, ETC.

from nltk import ne_chunk
from nltk.tokenize import word_tokenize
import nltk
def ner(text):
    words=word_tokenize(text)
    pos=pos_tag(words)
    ner=ne_chunk(pos)
    return ner

if __name__=="__main__":
    nltk.download('maxent_ne_chunker')
    nltk.download('words')
    text = "America, India, showing off the stop words filtration."
    print(ner(text))


# In[1]:


#UNDERSTANDING REGEX


# In[6]:


import re
def regex(text):
    pattern = re.compile(r'\d+')
    result = pattern.findall(text)
    return result

if __name__=="__main__":
    text = "This21 is a sample text, showing 33off the stop words filtration."
    print(regex(text))


# In[1]:


#REGEX FOR EMAIL

import re
def regex(text):
    pattern = re.compile(r'\S+@\S+')
    result = pattern.findall(text)
    return result

if __name__=="__main__":

    text = "This21 is a sample text, pmahule03@gmail.com showing 33off the stop words filtration."
    print(regex(text))


# In[11]:


#split the text into sentences
import re
def regex(text):
    a=re.split(r'\s', text)
    return a

if __name__=="__main__":
    text = "This21 is a sample text,showing 33off the stop words filtration."
    print(regex(text))


# In[12]:


#re.search
import re
def regex(text):
    pattern = re.compile(r'\d+')
    result = pattern.search(text)
    return result

if __name__=="__main__":
    text = "This21 is a sample text, showing 33off the stop words filtration."
    print(regex(text))


# In[19]:


#find pattern
pattern=["hello","world"]
text="hello worl"
text1="abc"
for i in pattern:
    print("Searching for {} in {}".format(i,text))
    if re.search(i,text):
        print("Matched")
    else:
        print("Not Matched")


# In[23]:


#frequency distribution
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
import nltk

def freq_dist(text):
    words=word_tokenize(text)
    fdist=FreqDist(words)
    return fdist

if __name__=="__main__":
    nltk.download('punkt')
    text = "This is a sample text, showing off the stop words filtration."
    print(freq_dist(text).plot())


# In[28]:


#implementing positiveNaiveBayesClassifier
import nltk
import random
from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import numpy as np
from nltk.classify import NaiveBayesClassifier
from nltk.classify import MaxentClassifier
from nltk.classify import DecisionTreeClassifier
from nltk.classify import SklearnClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

class VoteClassifier(ClassifierI):
    def __init__(self,*classifiers):
        self._classifiers=classifiers
    def classify(self,features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
        return mode(votes)
    def confidence(self,features):
        votes=[]
        for c in self._classifiers:
            v=c.classify(features)
            votes.append(v)
        choice_votes=votes.count(mode(votes))
        conf=choice_votes/len(votes)
        return conf

lemmatizer=WordNetLemmatizer()
stop_words=set(stopwords.words('english'))
def create_lexicon(pos,neg):
    lexicon=[]
    for fi in pos+neg:
        with open(fi,'r') as f:
            contents=f.readlines()
            for l in contents[:100]:
                all_words=word_tokenize(l.lower())
                lexicon+=list(all_words)
    lexicon=[lemmatizer.lemmatize(i) for i in lexicon]
    w_counts=FreqDist(lexicon)
    l2=[]
    for w in w_counts:
        if 1000>w_counts[w]>50:
            l2.append(w)
    return l2

def sample_handling(sample,lexicon,classification):
    featureset=[]
    with open(sample,'r') as f:
        contents=f.readlines()
        for l in contents[:100]:
            current_words=word_tokenize(l.lower())
            current_words=[lemmatizer.lemmatize(i) for i in current_words]
            features=np.zeros(len(lexicon))
            for word in current_words:
                if word.lower() in lexicon:
                    index_value=lexicon.index(word.lower())
                    features[index_value]+=1
            features=list(features)
            featureset.append([features,classification])
    return featureset

def create_feature_sets_and_labels(pos,neg,test_size=0.1):
    lexicon=create_lexicon(pos,neg)
    features=[]
    features+=sample_handling('pos.txt',lexicon,[1,0])
    features+=sample_handling('neg.txt',lexicon,[0,1])
    random.shuffle(features)
    features=np.array(features)
    testing_size=int(test_size*len(features))
    train_x=list(features[:,0][:-testing_size])
    train_y=list(features[:,1][:-testing_size])
    test_x=list(features[:,0][-testing_size:])
    test_y=list(features[:,1][-testing_size:])
    return train_x,train_y,test_x,test_y

if __name__=="__main__":
    train_x,train_y,test_x,test_y=create_feature_sets_and_labels('pos.txt','neg.txt')
    clf=NaiveBayesClassifier.train(train_x,train_y)
    clf2=MaxentClassifier.train(train_x,train_y)
    clf3=DecisionTreeClassifier.train(train_x,train_y)
    clf4=SklearnClassifier(SVC()).train(train_x,train_y)
    clf5=SklearnClassifier(LogisticRegression()).train(train_x,train_y)
    vote_clf=VoteClassifier(clf,clf2,clf3,clf4,clf5)
    print("Accuracy of Naive Bayes Classifier: ",(nltk.classify.accuracy(clf,test_x))*100)
    print("Accuracy of Max Entropy Classifier: ",(nltk.classify.accuracy(clf2,test_x))*100)
    print("Accuracy of Decision Tree Classifier: ",(nltk.classify.accuracy(clf3,test_x))*100)
    print("Accuracy of SVM Classifier: ",(nltk.classify.accuracy(clf4,test_x))*100)
    print("Accuracy of Logistic Regression Classifier: ",(nltk.classify.accuracy(clf5,test_x))*100)
    print("Accuracy of Voting Classifier: ",(nltk.classify.accuracy(vote_clf,test_x))*100)


# In[ ]:




