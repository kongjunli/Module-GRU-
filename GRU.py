# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 10:08:29 2017

@author: kjl
"""
import gensim
import numpy as np
import pandas as pd
import jieba
import mysql.connector
from gensim.models import word2vec
import logging
from sklearn.cross_validation import train_test_split
from gensim.models.word2vec import Word2Vec
from sklearn.linear_model import LinearRegression
from keras import metrics
import matplotlib.pyplot as plt
from keras.constraints import maxnorm
# custom R2-score metrics for keras backend
from keras import backend as K

from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import GRU,LSTM
from keras import regularizers
from keras.preprocessing import sequence
from keras import regularizers
import datetime
import numpy
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn import preprocessing
from keras.optimizers import SGD
from keras.utils import to_categorical
starttime = datetime.datetime.now()
#随机种子
seed =6868

class GetData():
    def Data(self):
        db=mysql.connector.connect(host='127.0.0.1',user='root',password='lkmn159357',database='spiderdb',charset='utf8')
        cursor=db.cursor()
        Content=[]
        sql='select day,authorscore,categoryscore,titlescore,titlepolarity,summaryscore,logscan from spiderdb.paper'              
        cursor.execute(sql)
        data=cursor.fetchall()
        return data   

obj=GetData()
datas=obj.Data() 
x=[]
y=[]
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
for i in range(len(datas)):
    try:
        x.append(datas[i][0:6])
        y.append(datas[i][6])
    except:
        continue 
for train_ratio in [0.9,0.6,0.3]:
    RMSE = [] #均方根误差
    MSE = []
    MAE = []
    R2 = []
    print "\n train_ratio:",train_ratio
    for i in range(5):
        print "-----------------The",i+1,"Experiment-------------"   
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_ratio,random_state=seed)
        #Normalization for train and test data set. http://scikit-learn.org/stable/modules/preprocessing.html
        #feature_range=(0, 1)
        print "min(y)",min(y)
        print "max(y)",max(y)
        print "original y train",y_train[0:10]
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0,1))  #1-3 10-12
        y_train=np.array(y_train).reshape(-1,1)
        y_train = min_max_scaler.fit_transform(y_train)
        y_test=np.array(y_test).reshape(-1,1)
        y_test = min_max_scaler.transform((y_test))
        model = Sequential()
        model.add(Embedding(10000,128))
        model.add(Dropout(0.2))
        model.add(GRU(128,dropout_U=0.2,kernel_regularizer=regularizers.l2(0.01)))
        model.add(Dense(1,W_regularizer=regularizers.l2(0.01)))
        model.add(Activation(' Leaky ReLU'))
        model.compile(loss='mse',optimizer='RMSprop',metrics=['mean_absolute_error','mse'])
        batch_size =2 #128,32,6,2 
        model.fit(x_train, y_train, batch_size=batch_size, nb_epoch=3,validation_data=(x_test,y_test))
        scores = model.evaluate(x_test, y_test, batch_size=batch_size)
        print "Model Summary",model.summary()
        print "MAE:%s: %.5f" % (model.metrics_names[1], scores[1]) #mae
        print "MSE:%s: %.5f" % (model.metrics_names[2], scores[2]) #mse
        probabilities = model.predict(x_test)
        print "Probabilities:",probabilities
        print "y_test",y_test
        print "pro shape",len(probabilities)
        print "y_test shape",len(y_test)
        print "mean absolure error:",(mean_absolute_error(y_test,probabilities))
        print "mean_squared_error:",(mean_squared_error(y_test,probabilities))
        print "RMSE:%.5f"%(np.sqrt(scores[2])) #rmse
        print "r2_score:",(r2_score(y_test,probabilities))
        RMSE.append(np.sqrt(scores[2]))
        MSE.append(mean_squared_error(y_test,probabilities))
        MAE.append(mean_absolute_error(y_test,probabilities))
        R2.append(r2_score(y_test,probabilities))
    print "Average RMSE:",np.mean(RMSE)
    print "Average MSE:",np.mean(MSE)    
    print "Average MAE:",np.mean(MAE)
    print "Average R2:",np.mean(R2)      
#figures(hist)  
#model.save('C:\Users\kjl\Desktop\paper\module')
endtime = datetime.datetime.now()
print (endtime - starttime).seconds
