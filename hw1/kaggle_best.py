import numpy as np
import csv
import pdb
import pandas
import sys



testfile_path=sys.argv[1]
outfile_path=sys.argv[2]

# read model
w = np.load('kaggle_best.npy')
std=np.load('std.npy')
mean=np.load('mean.npy')
bias=w[0]
w=np.delete(w,0)

feature_select=np.load('feature.npy')
test_x = []
n_row = 0
text = open(testfile_path ,"r")
row = csv.reader(text , delimiter= ",")
eps=1e-8


#select feature u want
feature_index=range(18)
for r in row:
    if n_row %18 == 0:
        test_x.append([])
    for i in range(2,11):
        if (n_row%18)in feature_index:
            if r[i] !="NR":
                test_x[n_row//18].append(float(r[i]))
            else:
                test_x[n_row//18].append(0)
    n_row = n_row+1

text.close()
test_x = np.array(test_x)

my_test=np.zeros((240,162))
for date in range(240):
    for hour in range(9):
        for feature in range(18):
            my_test[date,feature+hour*18]=test_x[date,feature*9+hour]


my_test=my_test[:,feature_select]


featureOrder=2
def predict(X):
	if featureOrder:
		tX = X
		for i in np.arange(2, featureOrder+1):
			tX = np.append(tX, X**i, axis=1)
		X = tX
	X = (X-mean)/(std+eps)
	return np.dot(np.array(X), w)+bias

out=predict(my_test)
outputfile = open(outfile_path, 'w')
csv_output = csv.writer(outputfile)

csv_output.writerow(['id', 'value'])

for idx, value in enumerate(out):
	csv_output.writerow(['id_'+str(idx), value])

outputfile.close()
