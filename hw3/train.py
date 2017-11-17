# This code is using tensorflow backend
#!/usr/bin/env python
# -- coding: utf-8 --

from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Flatten, Activation, Reshape, Concatenate, BatchNormalization,LeakyReLU
from keras.layers.convolutional import Conv2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.optimizers import SGD, Adam, Adadelta
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

import pdb
import os
import numpy as np
import argparse
import time
import pandas as pd
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.system('echo $CUDA_VISIBLE_DEVICES')

PATIENCE = 20 # The parameter is used for early stopping

classes = 7

def load_data(filename):
    df = pd.read_csv(filename)
    df_p = pd.DataFrame(df.feature.str.split().tolist()).astype(float).rename(columns = lambda x:'feature')
    df_p.insert(0, 'label', df.label)
    df_train, df_val = train_test_split(df_p, test_size=0.2)


    label_val = df_val['label']
    img_val = df_val['feature']

    label = df_train['label']
    img = df_train['feature']

    label_num = label.value_counts()
    ## make all category number the same
    for i in range(classes):
        df_append=df_train [df_train.label==i].sample(label_num.max()-label_num[i], replace=True)
        df_train = df_train.append(df_append)

    label = df_train['label']
    img = df_train['feature']


    one_hot_train = pd.get_dummies(label).astype(float)
    one_hot_train = one_hot_train.as_matrix()

    one_hot_val= pd.get_dummies(label_val).astype(float)
    one_hot_val = one_hot_val.as_matrix()

    img_train,_ = preprocess(img)
    img_val,_ = preprocess(img_val)

    img_train = img_train.as_matrix()
    img_val = img_val.as_matrix()

    return img_train ,one_hot_train, img_val, one_hot_val


def preprocess(X, mean=None , batch=False):
    Xn = X.astype(np.float32)
    Xn = Xn / 255.0
    if mean is None:
        mean = np.mean(Xn, axis=0)



    # Xn = Xn - mean
    return Xn, mean

def build_model():
    input_img = Input(shape=(48, 48, 1))


    block1 = Conv2D(64, kernel_size=(5, 5), input_shape=(48, 48, 1), padding='same', kernel_initializer='glorot_normal')(input_img)
    block1 = LeakyReLU(alpha=1./20)(block1)
    block1 = BatchNormalization()(block1)
    block1 = MaxPooling2D(pool_size=(2, 2), padding='same')(block1)
    block1 = Dropout(0.3)(block1)

    block2 = Conv2D(128, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal')(block1)
    block2 = LeakyReLU(alpha=1./20)(block2)
    block2 = BatchNormalization()(block2)
    block2 = MaxPooling2D(pool_size=(2, 2), padding='same')(block2)
    block2 = Dropout(0.3)(block2)

    block3 = Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal')(block2)
    block3 = LeakyReLU(alpha=1./20)(block3)
    block3 = BatchNormalization()(block3)


    block3 = AveragePooling2D(pool_size=(2, 2), padding='same')(block3)
    block3 = Dropout(0.4)(block3)



    block4 = Conv2D(512, kernel_size=(3, 3), padding='same', kernel_initializer='glorot_normal')(block3)
    block4 = LeakyReLU(alpha=1./20)(block4)
    block4 = BatchNormalization()(block4)
    block4 = AveragePooling2D(pool_size=(2, 2), padding='same')(block4)
    block4 = Dropout(0.4)(block4)
    block4 = Flatten()(block4)




    fc1 = Dense(1024, activation='relu')(block4)
    fc1 = BatchNormalization()(fc1)
    fc1 = Dropout(0.5)(fc1)


    fc2 = Dense(256, activation='relu')(fc1)
    fc2 = BatchNormalization()(fc2)
    fc2 = Dropout(0.5)(fc2)


    predict = Dense(7)(fc2)
    predict = Activation('softmax')(predict)

    model = Model(inputs=input_img, outputs=predict)
    opt = Adam(lr=1e-3)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model





def train(batch_size, num_epoch, pretrain, save_every, train_pixels, train_labels, val_pixels, val_labels):

    model = build_model()




    datagen = ImageDataGenerator( featurewise_center=False,
                                samplewise_center=False,
                                featurewise_std_normalization=False,
                                samplewise_std_normalization=False,
                                zca_whitening=False,
                                rotation_range=20.,
                                width_shift_range=0.2,
                                height_shift_range=0.2,
                                shear_range=0.2,
                                zoom_range=[0.8,1.2],
                                channel_shift_range=0.1,
                                fill_mode='nearest',
                                horizontal_flip=True,
                                vertical_flip=False,
                                preprocessing_function=None,
                                )
    # datagen.fit(train_pixels)
    num_epoch = 4000
    callbacks = []
    early_stopping_callback = EarlyStopping(monitor='val_acc', patience=200)
    check = ModelCheckpoint('./model-{epoch:05d}-{val_acc:.5f}.h5', monitor='val_acc', save_best_only=True, period=1)

    history=model.fit_generator(
            datagen.flow(train_pixels, train_labels, batch_size=batch_size),
            steps_per_epoch=len(train_pixels)//batch_size,
            epochs=num_epoch,
            validation_data=(val_pixels, val_labels),
            callbacks=[early_stopping_callback, check]
            )

    import matplotlib.pyplot as plt
    #summarize history for accuracy
    plt.figure()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Loss')
    plt.ylabel('Loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()





def main(argv):
    train_path = argv[1]

    train_path=train_path.replace('\r','')
    train_path=train_path.replace('\r\n','')
    train_pixels, train_labels, valid_pixels, valid_labels = load_data(train_path)


    # training data
    print ('# of training instances: ' + str(len(train_labels)))

    # validation data
    print ('# of validation instances: ' + str(len(valid_labels)))


    valid_pixels_res=[]
    train_pixels_res=[]

    for i in range(len(valid_labels)):
        valid_pixels_res.append(valid_pixels[i].reshape((48, 48, 1)))
    for i in range(len(train_labels)):
        train_pixels_res.append(train_pixels[i].reshape((48, 48, 1)))
    valid_pixels_res = np.asarray(valid_pixels_res)
    train_pixels_res = np.asarray(train_pixels_res)


    # start training
    train(128,4000,False, True,train_pixels_res, train_labels,valid_pixels_res, valid_labels,)


if __name__=='__main__':
    main(sys.argv)
