import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier
import pdb
import sys



# set pandas chained_assignment flag = None here
pd.options.mode.chained_assignment = None

training_file=sys.argv[1]
testing_file=sys.argv[2]
output_path=sys.argv[3]


def preprocess_features(dframe):
    for column in dframe:
        enc = LabelEncoder()
        if(column not in ['age','education_num','fnlwgt','capital_gain','capital_loss','hours_per_week']):
            dframe[column] = enc.fit_transform(dframe[column])
    return dframe

def data_read(train_file,test_file):
    # import data and preprocess
    df = pd.read_csv(train_file)

    # select and preprocess features
    le_data = LabelEncoder()
    features = ['age','workclass','education','marital_status','occupation','education_num','race','sex','relationship','capital_gain','capital_loss','hours_per_week','native_country','income']
    data = df[features]
    data = preprocess_features(data)

    # select target
    target = data['income']
    data = data.drop('income', axis=1)

    df2 = pd.read_csv(test_file)
    features = ['age','workclass','education','marital_status','occupation','education_num','race','sex','relationship','capital_gain','capital_loss','hours_per_week','native_country']
    in_test = df2[features]
    in_test = preprocess_features(in_test)

    # select target
    X_train,y_train=data,target
    X_test=in_test
    return X_train,X_test,y_train


[X_train, X_test, y_train]=data_read(training_file,testing_file)

print("training")
clf = GradientBoostingClassifier(loss='deviance', n_estimators=300, learning_rate=0.44,max_depth=2, random_state=0,max_features =6)

clf.fit(X_train, y_train)
predictions = clf.predict(X_test)

out = pd.DataFrame(predictions,columns =['label'])

output_path=output_path.replace('\r','')
output_path=output_path.replace('\r\n','')

out.index = out.index + 1
out.to_csv(output_path,index_label ='id')
