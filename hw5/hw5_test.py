import numpy as np
import pandas as pd
import keras.backend as K
import sys
from keras.models import load_model
from keras.engine.topology import Layer

# mean = 3.5817120860388076
# std = 1.116897661146206

mean = 0
std = 1

def Normalization(data):
    global mean ,std
    mean = np.mean(data)
    std  = np.std(data)
    return (data-mean)/std

def rmse(y_true, y_pred):
    y_pred = y_pred*std +mean  # for Normalization
    y_pred = K.clip(y_pred, 1.0, 5.0)
    return K.sqrt(K.mean(K.pow(y_true*std+mean - y_pred, 2)))

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true)))

def read_data(filename, user2id, movie2id):
    df = pd.read_csv(filename)
    df['UserID'] = df['UserID'].apply(lambda x: user2id[x])
    df['MovieID'] = df['MovieID'].apply(lambda x: movie2id[x])

    return df[['UserID', 'MovieID']].values



def main(argv):
    user2id = np.load('./model/user2id.npy')[()]
    movie2id = np.load('./model/movie2id.npy')[()]
    test_file = argv[1]
    out_file = argv[2]
    model_file = './model/8787_084784.h5'

    X_test = read_data(test_file, user2id, movie2id)



    model = load_model(model_file, custom_objects={'rmse': rmse})
    # model = load_model(model_file,custom_objects={'root_mean_squared_error': root_mean_squared_error})
    pred = model.predict([X_test[:, 0], X_test[:, 1]]).squeeze()
    pred = pred.clip(1.0, 5.0)

    out = pd.DataFrame(pred,columns =['Rating'])
    out.index = out.index+1
    out.to_csv(out_file,index_label ='TestDataID')



if __name__ == '__main__':

    main(sys.argv)
