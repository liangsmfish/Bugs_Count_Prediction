from sklearn.linear_model import LinearRegression
import csv
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import random


def load_data():
    
    #pathProg = 'C:\\Users\\hello\\Dropbox\\bug'
    temp = []
    time = []
    num  = []
    
    #file = open(pathProg+'bug.csv', 'r')
    file = open('bug.csv', 'r')
    csvCursor = csv.reader(file)
    for row in csvCursor:
        temp.append(row)
    
    data = len(temp)
    x = np.reshape(temp, (data*2, 1))     
    x = x.astype(np.float64) 
    
    
    for i in range(len(x)):
        if i % 2 == 0:
            time.append(x[i])
        else:
            num.append(x[i])
    t = time[-1]         
    X = np.array(time).reshape((-1,1))
    X = np.reshape(X, (data, 1))    
     
    Y = np.array(num).reshape((-1,1))
    Y = np.reshape(Y, (data, 1))   
    
    file.close()
    
    return X, Y,t


if __name__ == '__main__':
    
    X, Y ,t = load_data()
    
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size = 0.3, random_state = 9)
    
    lm= LinearRegression()
    lm.fit(Xtrain, Ytrain)
    xxfit = np.array([])    
    yyfit = np.array([])
    
    t=int(t)
    #load regression
    for i in range(0,t,1):
        xfit = i          
        yfit = lm.predict(xfit)    
        xxfit = np.append(xxfit, xfit)
        yyfit = np.append(yyfit, yfit)     
    
    #draw regression    
    plt.plot(xxfit, yyfit, 'r',linestyle='-',markersize=10)
    
    #draw train
    plt.scatter(Xtrain,Ytrain,s=10,label='Training data',c='b')
    plt.scatter(X,Y,s=10)
    
    #draw test
    plt.scatter(Xtest,Ytest,s=10,label='Testing data')    

    Trainguess = lm.predict(Xtrain)
    Testguess = lm.predict(Xtest)
    
    #draw predicted
    plt.scatter(Xtest, Testguess,s=10,label='Predicted result',c='r')
    
    plt.xlabel('Time(hours)')
    plt.ylabel('Number of bug(s)')
    # 預測某依時間的蟲個數
    
    name = input('請輸入時間(小時)：')
    name=int(name)
    to_be_predicted = np.array([name])
    
    to_be_predicted = to_be_predicted.reshape(1,-1)
    predicted = lm.predict(to_be_predicted)
    predicted = int(predicted)
    plt.legend(loc='best')
    
    
    
    print("預測粉蝨個數",predicted)
    Testguess= lm.predict(Xtest)
    
    #print("Training Error:",mean_squared_error(Ytrain, Trainguess))
    #print("Testing Error:",mean_squared_error(Ytest, Testguess))    
    
    