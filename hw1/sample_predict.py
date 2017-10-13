import numpy as np
import csv
import pdb
import pandas
import sys


testfile_path=sys.argv[1]
outfile_path=sys.argv[2]

# read model
w = np.load('baseline.npy')

test_x = []
n_row = 0
text = open(testfile_path ,"r")
row = csv.reader(text , delimiter= ",")


#select feature u want
feature_index=[9]
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




# add square term
test_x = np.concatenate((test_x,test_x**2), axis=1)

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)


ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append(a)

text = open(outfile_path, "w+")
s = csv.writer(text,delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i])
text.close()
