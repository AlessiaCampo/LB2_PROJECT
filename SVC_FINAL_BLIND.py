
import argparse
import os
from sklearn import svm
import pandas as pd
import numpy as np
import csv
import time
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
start_time=time.time()


parser=argparse.ArgumentParser()
parser.add_argument("-p")
parser.add_argument("-d")
parser.add_argument("-w")
parser.add_argument("-p_b")
parser.add_argument("-d_b")
args= parser.parse_args()


def class_ss(dssp):
    for line in dssp:
        if ">" not in line:
            #sec_str=list("".join(line[:-1]))
            line=line.rstrip()
            sec_str=list(line)
            ss=[]
            for el in range(len(sec_str)):
                if sec_str[el]=="H":
                    ss.append(1)
                elif sec_str[el]=="E":
                    ss.append(2)
                elif sec_str[el]=="C":
                    ss.append(3)
            return(ss)

def profile_as_array(profiles):
    list_of_rows=[]
    for rows in profiles:
        new_rows=rows.rstrip().split("\t")
        list_of_rows.append(new_rows[2:])
    list_aa=list_of_rows[0]
    list_of_counts=list_of_rows[1:]
    list_of_counts=[[float(j) for j in i]for i in list_of_counts]
    #return(list_of_counts, list_aa)
    w_size=int(args.w)//2
    for i in range(w_size):
        list_of_counts.append([0.0 for zero in range(20)])
        list_of_counts=[[0.0 for zero in range(20)]]+list_of_counts
    return(list_of_counts,list_aa,w_size)



#for filename in  profiles_l:
    #filename=filename.split(".")
    #filename=".".join(filename[:-3])

#sets=args.set_train
dssp_l=os.listdir(args.d)
#print(len(dssp_l))
profiles_l=os.listdir(args.p)
#set_path="./cv/"
#X_train=np.zeros()
X_train=[]
y_train=[]
for filename in profiles_l:
    for filename1 in dssp_l:
        if filename[:-19]==filename1[:-5]:
            dssp=open(args.d+filename1,"r")
            profiles=open(args.p+filename,"r")
            ss=class_ss(dssp) #for each sequence print the array in which are stored the observed classes
            y_train+=ss
            counts_profile, aa,w_size=profile_as_array(profiles) #for each sequence print the profiles with 0S AT THE END AND AT THE BEGINNING; the window size and the list of AA
            for pos in range(w_size, len(counts_profile)-w_size):  #BUILD THE SLIDING WINDOW to produce the array X
                window=np.array(counts_profile[pos-w_size:pos+w_size+1])
                window_vector=window.flatten()
                X_train.append(window_vector)


profile_l_b=os.listdir(args.p_b)
dssp_l_b=os.listdir(args.d_b)
#test_set=open(set_path+"set"+args.set_test)
y_test=[]
X_test=[]
for filename in profile_l_b:
    #print(filename[:-19])
    for filename1 in dssp_l_b:
        #print(filename1[:-5])
        if filename[:-19]==filename1[:-5]:
            dssp_test=open(args.d_b+filename1,"r")
            profiles_test=open(args.p_b+filename,"r")
            ss=class_ss(dssp_test)
            y_test+=ss
            test_counts_profile,aa,w_size=profile_as_array(profiles_test)
            for pos in range(w_size, len(test_counts_profile)-w_size):  #BUILD THE SLIDING WINDOW to produce the array X
                window=np.array(test_counts_profile[pos-w_size:pos+w_size+1])
                window_vector=window.flatten()
                X_test.append(window_vector)


print("X_train len:", len(X_train),len(y_train))
print("X_test_len:",len(X_test))
mySVC=OneVsRestClassifier(SVC(C=2,gamma=0.5),n_jobs=3)
mySVC.fit(X_train,y_train)

print("TRAINING COMPLETED")
#mySVC=svm.SVC(C=2,gamma=0.5)
#mySVC.fit(X_train, y_train)

#print("Best parameter set found on development set:")
#print()
#print(grid.best_params_)
print()
#print("Grid scores on development set:")
print()
#means=grid.cv_results_['mean_test_score']
#stds=grid.cv_results_['std_test_score']

print()
print("Detailed classification report:")
print()
print("The model is trained on the full development set.")
print("The scores are computed on the full evaluation set.")
print()
y_true, y_pred = y_test, mySVC.predict(X_test)
print(classification_report(y_true, y_pred))
print()
print(confusion_matrix(y_true, y_pred))


print("%s seconds" %(time.time()-start_time))
