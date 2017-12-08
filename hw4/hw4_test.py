from keras.models import load_model

import pdb
import pandas as pd
import numpy as np
import os
import sys
import tensorflow as tf
import keras.backend.tensorflow_backend as K

from util import DataManager


def main(argv):
    filename = argv[1]
    output_path= argv[2]
    output_path=output_path.replace('\r','')
    output_path=output_path.replace('\r\n','')
    dm = DataManager()
    dm.add_data('test_data',filename,False)
    dm.load_tokenizer('./model/token_25k.pk')
    dm.to_sequence(40)

    model = load_model('./model/00017-0.82720.h5')
    model.summary()


    val_proba = model.predict(dm.data['test_data'])
    val_classes = [1 if value>0.5 else 0 for value in val_proba]

    out = pd.DataFrame(val_classes,columns =['label'])
    out.to_csv(output_path,index_label ='id')






if __name__=='__main__':
    main(sys.argv)
