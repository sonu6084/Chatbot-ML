import re
import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from collections import Counter
import seaborn as sns
import dictionary
import matplotlib.pyplot as plt
from nltk.probability import FreqDist
import numpy as np
from restaurant_info import get_rest_info

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

df=pd.read_csv("chatbot_classes.csv")

ques_words=[]
list_sent=list(df['question'])

for sentence in list_sent:
    for words in sentence.split():
        ques_words.append(words)


fd=FreqDist()  
for word in ques_words:
    fd[word.lower()]+=1

labels,keys=zip(*fd.items())
labels=[]
keys=[]

for T in fd.most_common(10):
    labels.append(T[0])
    keys.append(T[1])

#function removing punctuations and lower the words
def lower_punc(text):
    w=[]
    for word in text.split():
        w.append(word.lower())
        
    wd=[]
    for word in w:
        for word in re.sub(r'[^\w]','',word).split():
            wd.append(word)
        
    return wd

from nltk.corpus import stopwords
sw=set(stopwords.words('english'))

#removing stopwords
def remove_stopwords(text):
    wd=[]
    for words in text:
        if words not in sw:
            wd.append(words)
            
    return wd

#negation handling
def negation_handle(a):
    wlist=[]
    negations=["no","not","cant","cannot","never","less","without","barely","hardly","rarely","no","not","noway","didnt","dont",'havent','couldnt','shouldnt','hasnt']
    counter=False
    for i,j in enumerate(a):
        if j in negations and i<len(a)-1:
            wlist.append(str(a[i]+'-'+a[i+1]))
            counter=True
        else:
            if counter is False:
                wlist.append(a[i])
            else:
                counter=False
        
    return wlist

#removing non-describtive words
from nltk.tag import pos_tag
def desc_words(words):
    mean_word=[]
    tags=['VB','VBP','VBD','VBG','VBN','JJ','JJR','JJS','RB','RBR','RBS','UH',"NN",'NNP'] 
    tag_words=pos_tag(words)
    for word in tag_words:
        if word[1] in tags:
            mean_word.append(word[0])

    return mean_word


#stemming

from nltk.stem.porter import PorterStemmer
st=PorterStemmer()
def stemming(words):
    stem_word=[]
    for word in words:
        stem_word.append(st.stem(word))
    return stem_word

#remaking of sentence
def remake(words):
    word=" "
    word=word.join(words)
    return word

#all data cleaning function called
def data_cleaning(words):
    data_wo_punc=lower_punc(words)
    words_wo_sw=remove_stopwords(data_wo_punc)
    neg_hand=negation_handle(words_wo_sw)
    word_describe=desc_words(neg_hand)
    word_stem=stemming(word_describe)
    new_words=remake(word_stem)
    return new_words
    
df['mod_ques']=df['question'].apply(data_cleaning)


#bag of words
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["question"]).toarray()

#tf-idf transformation
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
model = Pipeline([('vectoizer', CountVectorizer()),('tfidf', TfidfTransformer())])

X_train = model.fit_transform(df["mod_ques"]).toarray()


Y=df["answer"]
question="what is operating system "

from sklearn.naive_bayes import MultinomialNB
clf2 = MultinomialNB().fit(X_train, Y)

from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression().fit(X_train, Y)

from sklearn.tree import DecisionTreeClassifier
clf3 = DecisionTreeClassifier().fit(X_train, Y)


def Predict(text):
    P=model.transform([data_cleaning(text)])
    predict1=clf1.predict(P)
   
    predict2=clf2.predict(P)
    #print (predict2)
    
    predict3=clf3.predict(P)
    #print (predict3)
    
    final_predict=[]
    final_predict=list(predict1)+list(predict2)+list(predict3)
    #final_predict=list(predict3)
    final_predict = Counter(final_predict)
    #print ("Class of Question belongs to = ",final_predict.most_common(1)[0][0])
    
    return final_predict.most_common(1)[0][0]
    

import random

def generate_answer(predict_class):
    ans=random.choice(dictionary.tag_words[predict_class])
    return ans
    
def get_user_input(question):
#question = input("Enter Question =")
    prediction=Predict(question)
    if prediction=='restaurant':
        ans=prediction
    else:
        ans=generate_answer(prediction)
        #print('Bot > ',ans)
    return ans


