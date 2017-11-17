from keras.models import load_model

import pdb
import pandas as pd
import numpy as np
import os
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def preprocess(X, mean=None , batch=False):
    Xn = X.astype(np.float32)
    Xn = Xn / 255.0
    if mean is None:
        mean = np.mean(Xn, axis=0)



    # Xn = Xn - mean
    return Xn, mean


def main(argv):
    filename = argv[1]
    output_path= argv[2]
    output_path=output_path.replace('\r','')
    output_path=output_path.replace('\r\n','')
    df = pd.read_csv(filename)

    df_p = pd.DataFrame(df.feature.str.split().tolist()).astype(float).rename(columns = lambda x:'feature')
    df_p.insert(0, 'id', df.id)

    img_n,_ = preprocess(df_p['feature'].as_matrix())
    model = load_model('./example_GSP2_model-58.h5')
    model.summary()
    test_pixel=[]

    for i in range(len(img_n)):
        test_pixel.append(img_n[i].reshape((48, 48, 1)))
    test_pixel = np.array(test_pixel)

    val_proba = model.predict(test_pixel)
    val_classes = val_proba.argmax(axis=-1)

    out = pd.DataFrame(val_classes,columns =['label'])
    out.to_csv(output_path,index_label ='id')






if __name__=='__main__':
    main(sys.argv)
