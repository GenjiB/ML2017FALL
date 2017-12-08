import sys, argparse, os
import keras
import _pickle as pk
# import readline
import numpy as np

from keras import regularizers
from keras.models import Model
from keras.layers import Input, GRU, LSTM, Dense, Dropout, Bidirectional, Conv1D, MaxPooling1D, Flatten, Merge, LeakyReLU
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint

import keras.backend.tensorflow_backend as K
import tensorflow as tf
import pdb
from util import DataManager
from gensim.models import Word2Vec
# import matplotlib.pyplot as plt
from keras.models import load_model

parser = argparse.ArgumentParser(description='Sentiment classification')
parser.add_argument('-model', default= 'RCNN')
parser.add_argument('-action', default = 'train_corpus',choices=['train','test','semi','train_corpus'])

# training argument
parser.add_argument('--batch_size', default=128, type=float)
parser.add_argument('--nb_epoch', default=100, type=int)
parser.add_argument('--val_ratio', default=0.1, type=float)
parser.add_argument('--gpu_fraction', default=0.5, type=float)
parser.add_argument('--vocab_size', default=25000, type=int)
parser.add_argument('--max_length', default=40,type=int)

# model parameter
parser.add_argument('--loss_function', default='binary_crossentropy')
parser.add_argument('--cell', default='LSTM', choices=['LSTM','GRU','Conv'])
parser.add_argument('-emb_dim', '--embedding_dim', default=100, type=int)
parser.add_argument('-hid_siz', '--hidden_size', default=256, type=int)
parser.add_argument('--dropout_rate', default=0.4, type=float)
parser.add_argument('-lr','--learning_rate', default=0.001,type=float)
parser.add_argument('--threshold', default=0.02,type=float)

# output path for your prediction
parser.add_argument('--result_path', default='result.csv',)

# put model in the same directory
parser.add_argument('--load_model', default = None)
parser.add_argument('--save_dir', default = './model')

parser.add_argument('--train_data', default = None)
parser.add_argument('--semi_data', default = None)
args = parser.parse_args()

train_path = args.train_data
semi_path = args.semi_data

train_path=train_path.replace('\r','')
train_path=train_path.replace('\r\n','')

semi_path=semi_path.replace('\r','')
semi_path=semi_path.replace('\r\n','')

# train_path = './data/training_label.txt'
# semi_path = './data/training_nolabel.txt'

#124509
# build model
def simpleRNN(args,embedding_w):
    inputs = Input(shape=(args.max_length,)) #

    # Embedding layer
    embedding_inputs = Embedding(embedding_w.shape[0],
                                 embedding_w.shape[1],
                                 weights=[embedding_w],
                                 trainable=False)(inputs)

    # RNN
    return_sequence = False
    dropout_rate = args.dropout_rate
    if args.cell == 'GRU':
        RNN_cell = GRU(args.hidden_size,
                       return_sequences=return_sequence,
                       dropout=dropout_rate)
    elif args.cell == 'LSTM':
        RNN_cell = LSTM(args.hidden_size,
                        return_sequences=True,
                        dropout=dropout_rate)(embedding_inputs)
        RNN_cell = LSTM(args.hidden_size//4,
                        return_sequences=False,
                        dropout=dropout_rate)(RNN_cell)

        RNN_output = RNN_cell

    elif args.cell == 'Conv':
        RNN_cell = Conv1D(nb_filter=64, filter_length=4, border_mode='same')(embedding_inputs)
        RNN_cell = LeakyReLU(alpha=1./20)(RNN_cell)
        RNN_cell = MaxPooling1D(pool_length=2)(RNN_cell)
        RNN_cell = LSTM(args.hidden_size,
                        return_sequences=True,
                        dropout=dropout_rate)(RNN_cell)
        RNN_cell = LeakyReLU(alpha=1./20)(RNN_cell)
        RNN_cell = LSTM(args.hidden_size,
                        return_sequences=False,
                        dropout=dropout_rate)(RNN_cell)
        RNN_cell = LeakyReLU(alpha=1./20)(RNN_cell)
        RNN_output = RNN_cell

    # # DNN layer
    outputs = Dense(512,
                    kernel_regularizer=regularizers.l2(0.1))(RNN_output)
    outputs = LeakyReLU(alpha=1./20)(outputs)
    outputs = Dropout(dropout_rate)(outputs)
    outputs = Dense(1, activation='sigmoid')(outputs)


    model =  Model(inputs=inputs,outputs=outputs)

    # optimizer
    adam = Adam()
    print ('compile model...')

    # compile model
    model.compile( loss=args.loss_function, optimizer=adam, metrics=[ 'accuracy',])
    model.summary()
    return model

# def plot_figure(history):
#         #summarize history for accuracy
#         plt.figure()
#         plt.plot(history.history['acc'])
#         plt.plot(history.history['val_acc'])
#         plt.title('Accuracy')
#         plt.ylabel('Accuracy')
#         plt.xlabel('epoch')
#         plt.legend(['train', 'test'], loc='upper left')
#
#         # summarize history for loss
#         plt.figure()
#         plt.plot(history.history['loss'])
#         plt.plot(history.history['val_loss'])
#         plt.title('Loss')
#         plt.ylabel('Loss')
#         plt.xlabel('epoch')
#         plt.legend(['train', 'test'], loc='upper left')
#         plt.show()





def main():
    # limit gpu memory usage
    def get_session(gpu_fraction):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=gpu_fraction)
        return tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    K.set_session(get_session(args.gpu_fraction))

    save_path = os.path.join(args.save_dir,args.model)
    if args.load_model is not None:
        load_path = os.path.join(args.save_dir,args.load_model)

           #####read data#####
    dm = DataManager()
    print ('Loading data...')
    if args.action == 'train':
        dm.add_data('train_data', train_path, True)
    elif args.action == 'semi':
        dm.add_data('train_data', train_path, True)
        dm.add_data('semi_data', semi_path, False)
    elif args.action == 'train_corpus':
        dm.add_data('train_data', train_path, True)
        dm.add_data('semi_data', semi_path, False)
    else:
        raise Exception ('Implement your testing parser')


    # prepare tokenizer
    print ('get Tokenizer...')
    if args.load_model is not None:
        # read exist tokenizer
        dm.load_tokenizer(os.path.join(load_path,'token.pk'))
    else:
        # create tokenizer on new data
        dm.tokenize(args.vocab_size)

    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    if not os.path.exists('./model/token_25k.pk'):
        dm.save_tokenizer('./model/token_25k.pk')

    embedding_w = dm.get_vec_model('emb_1.npy',args.embedding_dim)
    dm.to_sequence(args.max_length)
        # initial model
    print ('initial model...')
    model = simpleRNN(args,embedding_w)
    model.summary()

    if args.load_model is not None:
        if args.action == 'train':
            print ('Warning : load a exist model and keep training')
        path = os.path.join(load_path,'model.h5')
        if os.path.exists(path):
            print ('load model from %s' % path)
            model.load_weights(path)
        else:
            raise ValueError("Can't find the file %s" %path)
    elif args.action == 'test':
        print ('Warning : testing without loading any model')

        # training
    if args.action == 'train_corpus':
        (X,Y),(X_val,Y_val) = dm.split_data('train_data', args.val_ratio)
        earlystopping = EarlyStopping(monitor='val_acc', patience = 3, verbose=1, mode='max')

        checkpoint = ModelCheckpoint(filepath='./model/'+'{epoch:05d}-{val_acc:.5f}.h5',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     monitor='val_acc',
                                     mode='max' )

        history = model.fit(X, Y,
                            validation_data=(X_val, Y_val),
                            epochs=args.nb_epoch,
                            batch_size=args.batch_size,
                            verbose=1,
                            shuffle= True,
                            callbacks=[checkpoint, earlystopping] )
        # plot_figure(history)
            # semi-supervised training
    elif args.action == 'semi':

        earlystopping = EarlyStopping(monitor='val_acc', patience = 10, verbose=1, mode='max')


        checkpoint = ModelCheckpoint(filepath='./model/semi/'+'{epoch:05d}-{val_acc:.5f}.h5',
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=False,
                                     monitor='val_acc',
                                     mode='max' )

        # repeat 10 times
        (X,Y),(X_val,Y_val) = dm.split_data('train_data', args.val_ratio)
        [semi_all_X] = dm.get_data('semi_data')
        semi_pred = model.predict(semi_all_X, batch_size=1024, verbose=True)
        dm.clean_data()
        dm.add_data('train_data', train_path, True)
        dm.add_data('test_data',test_path, False)
        dm.to_sequence(args.max_length)
        semi_X, semi_Y = dm.get_semi_data('test_data', semi_pred, args.threshold, args.loss_function)
        semi_X = np.concatenate((semi_X, X))
        semi_Y = np.concatenate((semi_Y, Y))
        print ('-- semi_data size: %d' %(len(semi_X)))

        model = simpleRNN(args,embedding_w)
        # train
        history = model.fit(semi_X, semi_Y,
                            validation_data=(X_val, Y_val),
                            epochs=40,
                            batch_size=args.batch_size,
                            callbacks=[checkpoint, earlystopping] )

        plot_figure(history)
if __name__ == '__main__':
        main()
