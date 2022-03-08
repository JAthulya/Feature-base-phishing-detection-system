import os
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.text import one_hot
import re
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from urllib.parse import urlparse, urlencode
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
import whois
import ipaddress
import csv

#PORT = int(os.environ.get('PORT'), 4567)

# df = pd.read_csv('finaldata.csv')
# x= df['url']
# y = df['label']

# voc_size = 10000
# messages = x.copy()

# corpus = []
# for i in range(0, len(messages)):
#     review = re.sub('[^a-zA-Z]',' ',urlparse(messages[i]).netloc)
#     review = review.lower()
#     review = review.split()
#     review=' '.join(review)
#     corpus.append(review)

# onehot_repr=[one_hot(words,voc_size)for words in corpus]
# sent_length = 50
# embedded_docs= pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)

# embedded_docs = np.array(embedded_docs)

# #x_final = np.array(embedded_docs)
# x_final = embedded_docs
# y_final  = np.array(y)
# from sklearn.model_selection import train_test_split
# x_train,x_test,y_train,y_test = train_test_split(x_final,y_final,test_size=0.20)


# #make the model and train it
# embedding_vector_features=100
# model = Sequential()
# model.add(Embedding(voc_size,embedding_vector_features,input_length=sent_length))
# model.add(LSTM(100))
# model.add(Dense(1,activation='sigmoid'))
# model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])


# model.fit(x_train,y_train,validation_data=(x_test,y_test),epochs=2,batch_size=64)

# y_pred1=model.predict(x_test) 
# classes_y1=np.round(y_pred1).astype(int)
# from sklearn.metrics import confusion_matrix
# confusion_n = confusion_matrix(y_test,classes_y1)
# from sklearn.metrics import accuracy_score
# print(accuracy_score(y_test, classes_y1))
# model.save("model1.h5")

def domaincreatedate(url):
    try:
        whois_info = whois.whois(url)
        cd = whois_info.get('creation_date')
        if whois_info.get('domain_name') == None:
            cd = 'No domain information for this URL'
        return cd
    except:
        return 'No informations about Domain creation date'

def ageofdomain(url):
    try:
        whois_info = whois.whois(url)
        ex = whois_info.get('expiration_date')
        cd = whois_info.get('creation_date')
        expiration_date = ex[0]
        creation_date = cd[0]
        days = abs((expiration_date - creation_date).days)
        if whois_info.get('domain_name')== None:
            days = 'No Domain informations about this URL'
        return days
    except :
        pass
    try:
        whois_info = whois.whois(url)
        ex = whois_info.get('expiration_date')
        cd = whois_info.get('creation_date')
        expiration_date = ex
        creation_date = cd
        days = abs((expiration_date - creation_date).days)
        if whois_info.get('domain_name')== None:
            days = 'No Domain informations about this URL'
        return days
    except:
        pass
    
    try:
        whois_info = whois.whois(url)
        ex = whois_info.get('expiration_date')
        cd = whois_info.get('creation_date')
        expiration_date = ex
        creation_date = cd[0]
        days = abs((expiration_date - creation_date).days)
        if whois_info.get('domain_name')== None:
            days = 'No Domain informations about this URL'
        return days
    except:
        pass
    try:
        whois_info = whois.whois(url)
        ex = whois_info.get('expiration_date')
        cd = whois_info.get('creation_date')
        expiration_date = ex[0]
        creation_date = cd
        days = abs((expiration_date - creation_date).days)
        if whois_info.get('domain_name')== None:
            days = 'No Domain informations about this URL'
        return days
    except:
        return 'No Domain information about this URL'
   

app = Flask(__name__)
model = tf.keras.models.load_model('model1.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    voc_size = 10000
    corpus=[]
    for i in request.form.values():
        
    #messages = [str(x) for x in request.form.values()]
        messages = i

        review = re.sub('[^a-zA-Z]',' ',urlparse(messages).netloc)
        review = review.lower()
        review = review.split()
        review=' '.join(review)
        corpus.append(review)
        onehot_repr=[one_hot(words,voc_size)for words in corpus]
        print(onehot_repr)
        sent_length = 50
        embedded_docs= pad_sequences(onehot_repr,padding='pre',maxlen=sent_length)
        x_test = embedded_docs
        y_pred = model.predict(x_test)
        classes_y=np.round(y_pred).astype(int)
        createdDate = domaincreatedate(messages)
        domainAge = ageofdomain(messages)
        
        header = ['url']
        data = [messages]
        with open('test.csv','w', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(data)
            f.close()
        
        df = pd.read_csv('test.csv')
        def havingIP(url):
          try:
            ipaddress.ip_address(url)
            ip = 1
          except:
            ip = 0
          return ip


        list=[]
        for i in df.url:
            ip = havingIP(i)
            list.append(ip)
        df['Have_IP']=list
        df.to_csv('test1.csv',index=False)
        #feature based detection
    
        
        
        
        
        
        
        #return render_template('index.html', prediction_text='url prediction -{}'.format(classes_y))
    return render_template('index.html',prediction_text=format(y_pred),prediction_text1=format(classes_y),created_date=createdDate
                           ,domain_age = domainAge)
           



if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8000, debug=True)
    
    
#app.run(host='0.0.0.0',port=8000, debug=True)

