import csv
import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
#from lecture7 import FeedforwardNN
import warnings
warnings.filterwarnings("ignore")
np.random.seed(0)

def make_data():
    #pathProg = 'C:\\Users\\hello\\Desktop\\bug'
    
    #pathProg = 'C:\\Users\\hello\\Dropbox\\bug'
    temp = []
    time=[]
    num=[]
   # file = open(pathProg + '\\bug.csv', 'r')
    file = open('bug.csv', 'r')
    csvCursor = csv.reader(file)
    for row in csvCursor:
        temp.append(row)
        
    data = len(temp)  #row 數量 
   
    x = np.reshape(temp, (data*2, 1)) #col1,2
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
    
    #print(X)
    #print(Y)    
    file.close()
    return X, Y,t

if __name__ == '__main__':
    X, Y ,t = make_data()
    #plt.scatter(X, Y)
    Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=9)
    #plt.plot(Xtest, 	Ytest, '.')
    mlp = MLPRegressor(
    hidden_layer_sizes=(5,),  activation='tanh', solver='lbfgs', alpha=0.001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.01, power_t=0.5, max_iter=10000, shuffle=True,
    random_state=9, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    mlp.fit(Xtrain, Ytrain)
    t=int(t)
    xxfit = np.array([])
    yyfit = np.array([])    
    
    for i in range(0,t,1):
        xfit = i          
        yfit = mlp.predict(xfit)    
        xxfit = np.append(xxfit, xfit)
        yyfit = np.append(yyfit, yfit)    
    
    plt.plot(xxfit, yyfit, 'r',linestyle='-',markersize=10)  
    plt.scatter(Xtrain,Ytrain,s=10,label='Training data')
    plt.scatter(Xtest,Ytest,s=10,label='Testing data')   
    
    
    plt.xlabel('Time(hours)')
    plt.ylabel('Number of bug(s)')
    
    # 預測某依時間的蟲個數    
    name = input('請輸入時間(小時)：')
    name= int(name)
    
    to_be_predicted = np.array([name])
     
    to_be_predicted = to_be_predicted.reshape(1,-1)
    predicted = mlp.predict(to_be_predicted)
    predicted = int(predicted)
    plt.legend(loc='best')    
    
    print("預測粉蝨個數",predicted)
    Testguess= mlp.predict(Xtest)
    Trainguess = mlp.predict(Xtrain)
    
    plt.scatter(Xtest, Testguess,s=10,label='Predicted result',c='r')
    #print("Training Error:",mean_squared_error(Ytrain, Trainguess))
    #print("Testing Error:",mean_squared_error(Ytest, Testguess))   

