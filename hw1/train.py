import csv
import numpy as np
from numpy.linalg import inv
import random
import math
import sys
import pdb
from sklearn.model_selection import train_test_split

def load_file_data(filepath):
    data = []
    # 每一個維度儲存一種污染物的資訊
    for i in range(18):
    	data.append([])

    n_row = 0
    text = open(filepath, 'r', encoding='big5')
    row = csv.reader(text , delimiter=",")
    for r in row:
        # 第0列沒有資訊
        if n_row != 0:
            # 每一列只有第3-27格有值(1天內24小時的數值)
            for i in range(3,27):
                if r[i] != "NR":
                    data[(n_row-1)%18].append(float(r[i]))
                else:
                    data[(n_row-1)%18].append(float(0))
        n_row = n_row+1
    text.close()

    x = []
    y = []
    # 每 12 個月
    data=np.array(data)

    #2,5,7,8,9,12
    feature_index=[9]
    x=data[feature_index,0:9].reshape(1,len(feature_index)*9, order='F') #column major
    for month in range(12):
        for i in range(480-9):
            tmp=data[feature_index,i+month*480:i+month*480+9].reshape(1,len(feature_index)*9, order='F')
            y.append(data[9,i+month*480+9])
            x=np.vstack((x,tmp))
    x = np.delete(x, (0), axis=0)

    # for i in range(12):
    #     # 一個月取連續10小時的data可以有471筆
    #     for j in range(471):
    #         x.append([])
    #         # 18種污染物
    #         # for t in range(18):
    #             # 連續9小時
    #         for s in range(9):
    #             # .reshape(1,27, order='F')
    #
    #             x[471*i+j].append(data[9,480*i+j+s] )
    #             # x[471*i+j].append(data[9][480*i+j+s] )
    #         y.append(data[9][480*i+j+9])
    x = np.array(x)
    y = np.array(y)

    # add square term
    x = np.concatenate((x,x**2), axis=1)

    # add bias
    x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
    return x_train, x_test, y_train, y_test
def train(x_train,y_train,x_test,y_test):
    n,d=x_train.shape
    lamda=0

    weight = np.zeros(d)
    l_rate = 10
    repeat = 10000

    s_gra = np.zeros(d)


    for i in range(repeat):
        hypo = np.dot(x_train,weight)
        loss = hypo - y_train
        gra = np.dot(x_train.transpose(),loss)+lamda*weight
        s_gra += gra**2
        ada = np.sqrt(s_gra)
        weight = weight - l_rate * gra/ada
        if i%1000==0:
            cost = np.sum(loss**2)/n
            cost=cost+lamda*sum(weight**2)/2*n
            cost_a  = math.sqrt(cost)

            val_cost = np.sum((np.dot(x_test,weight)-y_test)**2) / y_test.shape[0]
            val_cost=cost+lamda*sum(weight**2)/2*n
            val_cost_a  = math.sqrt(cost)
            print ('iteration: %d | Cost: %f  ' % ( i/1000,cost_a),val_cost_a)
    return weight



def main(argv):
    x_train, x_test, y_train, y_test=load_file_data('./train.csv')
    weight=train(x_train,y_train,x_test,y_test)


    # print("Linear train R^2: %f" % (sklearn.metrics.r2_score(y_train, y_hat)))
    # y_hat = predict(X_test_scale, theta)
    # print("Linear test R^2: %f" % (sklearn.metrics.r2_score(y_test, y_hat)))

    # save model
    np.save('model_only_pm25.npy',weight)
    print ("hellow ")


if __name__ == "__main__":
    main(sys.argv)
