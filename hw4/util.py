import os
import tensorflow as tf
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from gensim.models import Word2Vec
import _pickle as pk
import pdb
import re
from sklearn.model_selection import train_test_split


class DataManager:
    def __init__(self):
        self.data = {}
        self.vec_sentence=[]
        self.vec_model={}
        self.tokenizer={}
        self.vocab_size=0
    # Read data from data_path
    #  name       : string, name of data
    #  with_label : bool, read data with label or without label
    def clean_data(self):
        self.data={}
    def add_data(self,name, data_path, with_label=True):
        print ('read data from %s...'%data_path)
        X, Y = [], []
        with open(data_path,'r', encoding = 'utf8') as f:

            #skip first line : id text
            if name=='test_data':
                next(f)
            for line in f:
                if with_label:
                    line = line.split('+++$+++',1)
                    line[1] = re.sub('[\r,\n]','',line[1])

                    X.append(line[1])
                    Y.append(int(line[0]))

                    tmp= line[1].split(' ')
                    tmp = list(filter(None,tmp))
                    self.vec_sentence.append(tmp)
                elif name=='test_data':
                    id_x, line = line.split(",",1)
                    line = re.sub('[\r,\n]','',line)
                    X.append(line)

                    tmp = line.split(' ')
                    tmp = list(filter(None,tmp))
                    self.vec_sentence.append(tmp)
                elif name=='semi_data':
                    line = re.sub('[\r,\n]','',line)
                    X.append(line)
                    tmp = line.split(' ')
                    tmp = list(filter(None,tmp))
                    self.vec_sentence.append(tmp)

        if with_label:
            self.data[name] = [X,Y]
        else:
            self.data[name] = [X]


    # Build dictionary
    #  vocab_size : maximum number of word in yout dictionary
    def tokenize(self, vocab_size):
        print ('create new tokenizer')
        self.vocab_size=vocab_size
        self.tokenizer = Tokenizer(num_words=vocab_size,split=' ',filters='\n') #
        for key in self.data:
            print ('tokenizing %s'%key)
            texts = self.data[key][0]
            self.tokenizer.fit_on_texts(texts)
    # Save tokenizer to specified path
    def save_tokenizer(self, path):
        print ('save tokenizer to %s'%path)
        pk.dump(self.tokenizer, open(path, 'wb'))
    def word2vector(self,embedding_size):
        model = Word2Vec(self.vec_sentence, size = embedding_size, min_count = 6)
        self.vec_model = model
    def get_vec_model(self,path,embedding_size):
        if path is None:
            self.word2vector(embedding_size)
            # self.load_tokenizer('./model/RCNN/token.pk')
            vocab_size = self.tokenizer.num_words
            embedding = np.zeros((vocab_size, embedding_size))
            for word, i in self.tokenizer.word_index.items():
                if i==vocab_size:
                    break
                if word in self.vec_model:
                    vector = self.vec_model[word]
                    if vector is not None:
                        embedding[i] = vector
                else:
                    embedding[i] = embedding[0]
            np.save('./emb_1.npy',embedding)
        else:
            embedding = np.load(path)
        return embedding

    # Load tokenizer from specified path
    def load_tokenizer(self,path):
        print ('Load tokenizer from %s'%path)
        self.tokenizer = pk.load(open(path, 'rb'))

        # Convert words in data to index and pad to equal size
    #  maxlen : max length after padding
    def to_sequence(self, maxlen):
        self.maxlen = maxlen
        for key in self.data:
            print ('Converting %s to sequences'%key)
            tmp = self.tokenizer.texts_to_sequences(self.data[key][0])
            self.data[key][0] = np.array(pad_sequences(tmp, maxlen=maxlen,padding='post'))

    # Convert texts in data to BOW feature
    def to_bow(self):
        for key in self.data:
            print ('Converting %s to tfidf'%key)
            self.data[key][0] = self.tokenizer.texts_to_matrix(self.data[key][0],mode='count')

    # Convert label to category type, call this function if use categorical loss
    def to_category(self):
        for key in self.data:
            if len(self.data[key]) == 2:
                self.data[key][1] = np.array(to_categorical(self.data[key][1]))

    def get_semi_data(self,name,label,threshold,loss_function) :
        # if th==0.3, will pick label>0.7 and label<0.3
        label = np.squeeze(label)
        index = (label>1-threshold) + (label<threshold)
        semi_X = self.data[name][0]
        semi_Y = np.greater(label, 0.5).astype(np.int32)
        if loss_function=='binary_crossentropy':
            return semi_X[index,:], semi_Y[index]
        elif loss_function=='categorical_crossentropy':
            return semi_X[index,:], to_categorical(semi_Y[index])
        else :
            raise Exception('Unknown loss function : %s'%loss_function)

    # get data by name
    def get_data(self,name):
        return self.data[name]

    # split data to two part by a specified ratio
    #  name  : string, same as add_data
    #  ratio : float, ratio to split
    def split_data(self, name, ratio):
        data = self.data[name]
        X = data[0]
        Y = data[1]
        data_size = len(X)
        val_size = int(data_size * ratio)
        return (X[val_size:],Y[val_size:]),(X[:val_size],Y[:val_size])
