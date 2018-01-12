from keras.models import load_model,Model
from keras.layers import Input
from sklearn.cluster import KMeans
import os, sys
import pandas as pd
import numpy as np


def main(argv):
    img_file = argv[1]
    test_case = argv[2]
    predict_file = argv[3]


    X =np.load(img_file)
    X = X.astype('float32') / 255.
    X = np.reshape(X, (len(X), -1))

    model_path = './model/encoder.h5'
    encoder = load_model(model_path)
    encoded_imgs = encoder.predict(X)
    encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0], -1)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs)


    f = pd.read_csv(test_case)
    IDs, idx1, idx2 = np.array(f['ID']), np.array(f['image1_index']), np.array(f['image2_index'])

    o = open(predict_file, 'w')
    o.write("ID,Ans\n")
    for idx, i1, i2 in zip(IDs, idx1, idx2):
        p1 = kmeans.labels_[i1]
        p2 = kmeans.labels_[i2]
        if p1 == p2:
            pred = 1  # two images in same cluster
        else:
            pred = 0  # two images not in same cluster
        o.write("{},{}\n".format(idx, pred))
    o.close()


if __name__ == '__main__':

    main(sys.argv)
