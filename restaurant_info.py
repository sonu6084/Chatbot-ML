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
import sqlite3
conn=sqlite3.Connection('Restaurant_India.db')
db_df=pd.read_sql_query("SELECT * from zominfo;",conn )

from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

def low(words):
    return words.lower()

df=pd.read_csv("restaurant_classes.csv")

ques_words=[]
list_sent=list(df['question'])

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

def remove_stopwords(text):
    wd=[]
    for words in text:
        if words not in sw:
            wd.append(words)
            
    return wd

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
 
from nltk.tag import pos_tag
def desc_words(words):
    mean_word=[]
    tags=['VB','VBP','VBD','VBG','VBN','JJ','JJR','JJS','RB','RBR','RBS','UH',"NN",'NNP'] 
    tag_words=pos_tag(words)
    for word in tag_words:
        if word[1] in tags:
            mean_word.append(word[0])

    return mean_word

from nltk.stem.porter import PorterStemmer
st=PorterStemmer()
def stemming(words):
    stem_word=[]
    for word in words:
        stem_word.append(st.stem(word))
    return stem_word


def remake(words):
    word=" "
    word=word.join(words)
    return word

def data_cleaning(words):
    data_wo_punc=lower_punc(words)
    data_wo_sw=remove_stopwords(data_wo_punc)
    neg_hand=negation_handle(data_wo_sw)
    word_describe=desc_words(neg_hand)
    word_stem=stemming(word_describe)
    new_words=remake(word_stem)
    return new_words
    
df['mod_ques']=df['question'].apply(data_cleaning)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(df["question"]).toarray()

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
rest_ans_model = Pipeline([('cv', CountVectorizer()),
 ('tfidf', TfidfTransformer())])

X_train = rest_ans_model.fit_transform(df["mod_ques"]).toarray()
Y=df["answer"]
question="where which place the restaurant lies"

from sklearn.linear_model import LogisticRegression
clf1 = LogisticRegression().fit(X_train, Y)


P=rest_ans_model.transform([data_cleaning(question)])
#print(P.toarray())
predict1=clf1.predict(P.toarray())
#print (predict1)

from sklearn.naive_bayes import MultinomialNB
clf2 = MultinomialNB().fit(X_train, Y)

P=rest_ans_model.transform([data_cleaning(question)])
predict2=clf2.predict(P.toarray())
#print (predict2)

from sklearn.tree import DecisionTreeClassifier
clf3 = DecisionTreeClassifier().fit(X_train, Y)

P=rest_ans_model.transform([data_cleaning(question)])
predict3=clf3.predict(P)

def Predict(text):
    P=rest_ans_model.transform([data_cleaning(text)])
    predict1=clf1.predict(P)
   
    predict2=clf2.predict(P)
    #print (predict2)
    
    predict3=clf3.predict(P)
    #print (predict3)
    
    final_predict=[]
    final_predict=list(predict1)+list(predict2)+list(predict3)
    final_predict = Counter(final_predict)
    #print ("Class of Question belongs to = ",final_predict.most_common(1)[0][0])
    
    return final_predict.most_common(1)[0][0]
    
#ans=Predict('where which place the restaurant lies')

def get_rest_info(final_list):
    location=final_list[0]
    result_df=db_df[db_df['Locality'].apply(low)==location.lower()]
    if final_list[1].lower()=='all':
        result_df=result_df
    else:
        result_df=result_df[result_df['Restaurant_Name'].apply(low)==final_list[1].lower()]

    prediction=Predict(final_list[2])
    a=prediction
    return result_df[['Restaurant_Name',a]]


