# -*- coding:utf-8 -*-
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
from sklearn.preprocessing import scale
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Embedding
from keras.layers import LSTM
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
class GetData():
    def Data(self):
        db=mysql.connector.connect(host='127.0.0.1',user='root',password='lkmn159357',database='spiderdb',charset='utf8')
        cursor=db.cursor()
        Content=[]
        sql='select title,summary,pub_time,Author,category,logscan from spiderdb.paper'              
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
        x.append(datas[i][0:5])
        y.append(datas[i][5])
    except:
        continue




    
    
#随机种子
seed = 6
epochs = 50  #default:10 50
batch_size = 128
for train_ratio in [0.9,0.8,0.6,0.3]:
    RMSE = [] #均方根误差
    MSE = []
    MAE = []
    R2 = []
    print "\n train_ratio:",train_ratio
    for i in range(5):
        print "-----------------The",i+1,"Experiment-------------"    
        x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=train_ratio,random_state=seed+i-1)
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(11, 15))  #1-3 10-12 调节y值范围，缩小范围可以提高模型精度
        y_train=np.array(y_train).reshape(-1,1)
        y_train = min_max_scaler.fit_transform(y_train)
        y_test=np.array(y_test).reshape(-1,1)
        y_test = min_max_scaler.transform((y_test))
        
        n_dim=100  #300
        news_w2v = gensim.models.Word2Vec(x_train, min_count=10,size=n_dim)
        news_w2v.save('news_w2v2')
        news_w2v = gensim.models.Word2Vec.load('news_w2v2')
        news_w2v.train(x_train)
        #Build word vector for training set by using the average value of all word vectors in the tweet, then scale
        def buildWordVector(text, size):
            vec = np.zeros(size).reshape((1, size))
            count = 0
            for word in text:
                try:
                    vec += news_w2v[word].reshape((1, size))
                    count += 1.
                except KeyError:
                    continue
            if count != 0:
                vec /= count
            return vec
        
        
        train_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_train])
        train_vecs = scale(train_vecs)
        #Build test news vectors then scale
        test_vecs = np.concatenate([buildWordVector(z, n_dim) for z in x_test])
        test_vecs = scale(test_vecs)
            
        
        # 建立模型
        model = Sequential()
        model.add(Embedding(1000, 256))
        model.add(LSTM(128))
        model.add(Dropout(0.25))
        model.add(Dense(1))
        model.add(Activation('relu'))
        model.compile(loss='mse',
            optimizer='sgd',
            metrics=['mean_absolute_error','mse'])
    
        model.fit(train_vecs, y_train, batch_size=batch_size, nb_epoch=epochs,validation_data=(test_vecs,y_test))
        scores=model.evaluate(test_vecs, y_test, batch_size=batch_size)
        probabilities = model.predict(test_vecs)
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
    res = {"rmse":RMSE,"MSE":MSE,"MAE":MAE,"R2":R2,"Average RMSE":np.mean(RMSE),"Average MSE":np.mean(MSE),"Average MAE":np.mean(MAE),"Average R2":np.mean(R2)}
    res_name="Res_new_w2v_"+"epochs"+str(epochs)+"_batch_size"+str(batch_size)+"_train_ratio"+str(train_ratio)
    np.save("./res/"+res_name+".npy",res)
