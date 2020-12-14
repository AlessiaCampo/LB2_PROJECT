from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import classification_report,confusion_matrix
#from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import matthews_corrcoef, make_scorer
import os
import joblib
import argparse
import statistics
import numpy as np
import math
import time
import multiprocessing
from sklearn.model_selection import PredefinedSplit, cross_validate,cross_val_predict

start_time=time.time()



#-------------------------------------------------------------------------------

cv_path="/home/alessia/LAB2/project/PROJECTWORK/JUPYTER_PROJECT/profile_dssp_GOR_training/cv/"
profile_path="/home/alessia/LAB2/project/PROJECTWORK/JUPYTER_PROJECT/profile_dssp_GOR_training/profile_training/" #take the profiles of the training set
dssp_path="/home/alessia/LAB2/project/PROJECTWORK/JUPYTER_PROJECT/profile_dssp_GOR_training/dssp_mod/" #take the dssp of the training
#take also the dssp and the profiles of the blind set
#-------------------------------------------------------------------------------

#Preparation of the input ------------------------------------------------------
def input_SVM(filename,profile_path,dssp_path,pssm_files):
    window=17
    def profile_as_array(profile):
        list_of_rows=[]
        for rows in profile:
            new_rows=rows.rstrip().split("\t")
            list_of_rows.append(new_rows[2:])
        #list_aa=list_of_rows[0]
        list_of_counts=list_of_rows[1:]
        list_of_counts=[[float(j) for j in i]for i in list_of_counts]
        #return(list_of_counts, list_aa)
        #w_size=int(args.w)//2
        for i in range(window//2):
            list_of_counts.append([0.0 for zero in range(20)])
            list_of_counts=[[0.0 for zero in range(20)]]+list_of_counts
        return(list_of_counts)
    def class_ss(dssp):
        for line in dssp:
            if ">" not in line:
                line=line.rstrip()
                sec_str=list("".join(line))
                #print(len(sec_str))
                ss=[]
                for el in range(len(sec_str)):
                    if sec_str[el]=="H":
                        ss.append(1)
                    elif sec_str[el]=="E":
                        ss.append(2)
                    elif sec_str[el]=="C":
                        ss.append(3)
                return(ss)




    #Main loop
    X=[]
    y=[]

    if filename  in pssm_files:

        profile=open(profile_path+filename+".fasta.pssm.profile")
        dssp=open(dssp_path+filename+".dssp")
        profile_list=profile_as_array(profile)

        ss_class=class_ss(dssp)
        for el in range(len(ss_class)):
            window_list=profile_list[el:el+window]
            window_list=[el for li in window_list for el in li]
            y.append(ss_class[el])
            X.append(window_list)
        return(X,y)
    else:
        return([],[])

pssm_files=[] #In this this list there will be all the IDs of the Jpred profile that PSIBLAST have calculated
for pssm_file in os.listdir(profile_path):
    pssm_files.append(pssm_file[:-19])

X_list=[]
y_list=[]
fold_set=[]
for filename in os.listdir(cv_path):
    fold=filename[3]
    cv_file=open(cv_path+filename)
    for line in cv_file:
        ID=line.rstrip()
        #print(ID)
        X_train_test,y_train_test=input_SVM(ID,profile_path,dssp_path,pssm_files)
        if len(X_train_test)!=0:
            fold_set+=[int(fold) for i in range(len(y_train_test))]
            X_list=X_list+X_train_test
            y_list=y_list+y_train_test
X=np.asarray(X_list)
y=np.asarray(y_list)



def build_2_matrix(m3):
	m2_H=np.array([[0,0],[0,0]])
	m2_E=np.array([[0,0],[0,0]])
	m2_C=np.array([[0,0],[0,0]])

	#2 classes H
	m2_H[0][0]=m3[0][0]
	m2_H[0][1]=m3[0][1]+m3[0][2]
	m2_H[1][0]=m3[1][0]+m3[2][0]
	m2_H[1][1]=m3[1][1]+m3[1][2]+m3[2][1]+m3[2][2]

	#2 classes E
	m2_E[0][0]=m3[1][1]
	m2_E[0][1]=m3[1][0]+m3[1][2]
	m2_E[1][0]=m3[0][1]+m3[2][1]
	m2_E[1][1]=m3[0][0]+m3[0][2]+m3[2][0]+m3[2][2]

	#2 classes C
	m2_C[0][0]=m3[2][2]
	m2_C[0][1]=m3[2][0]+m3[2][1]
	m2_C[1][0]=m3[0][2]+m3[1][2]
	m2_C[1][1]=m3[0][0]+m3[0][1]+m3[1][0]+m3[1][1]

	return m2_H,m2_E,m2_C

def mcc(m):
  d=(m[0][0]+m[1][0])*(m[0][0]+m[0][1])*(m[1][1]+m[1][0])*(m[1][1]+m[0][1])
  return (m[0][0]*m[1][1]-m[0][1]*m[1][0])/math.sqrt(d)

def get_acc(cm):
    #print (sum(cm[0])+sum(cm[1])+sum(cm[2]))
	return float(cm[0][0]+cm[1][1]+cm[2][2])/(sum(cm[0])+sum(cm[1])+sum(cm[2]))
def sen(m):
	return m[0][0]/(sum(m[0]))

def ppv(m):
	return m[0][0]/(m[0][0]+m[1][0])



#my_scorer= make_scorer(myMCC)
ps=PredefinedSplit(fold_set)
#print(ps)
print("NUMBER OF K-FOLDS=", ps.get_n_splits()) #returns the number  of k-folds
print("\nThe lenght of the set X:", len(X), "y: ", len(y))
print("\nThe lenght of the fold set number:", len(fold_set))
print()
print(len(X))
#print(len(X[-1]))
#print("y_true")
print(len(y))
print(type(y[1]))


mySVC = SVC(C=2.0, kernel='rbf', gamma=0.5) #build the model SVC
print()
print("TRAINING INITIALIZATION")
print()
y_pred = cross_val_predict(mySVC, X, y, cv=ps, n_jobs=2)
print()
print("PREDICTION COMPLETED")
print()
print("RESULTS:")
cm = confusion_matrix(y, y_pred,labels=[1,2,3])
#print("y_pred")
#print(y_pred)

print("Confusion Matrix 3x3:")
print(cm)
print()
print("Total Average Accuracy:",get_acc(cm))
print()
m2_H,m2_E,m2_C=build_2_matrix(cm)
print()
print("PARTIAL MATRICES 2x2:")
print()
print("CM C")
print(m2_C)
print()
print("CM H")
print(m2_H)
print()
print("CM E")
print(m2_E)
print()
print()
print("AVERAGE CORRELATION COEFFICIENTS:")
print("mcc C:",mcc(m2_C),"mcc H:",mcc(m2_H),"mcc E:",mcc(m2_E))
print()
print("AVERAGE SENSITIVITY/RECALL:")
print("sen C:",sen(m2_C),"sen H:",sen(m2_H),"sen E:",sen(m2_E))
print()
print("AVERAGE PPV/PRECISION")
print("ppv C:",ppv(m2_C),"ppv H:",ppv(m2_H),"ppv E:",ppv(m2_E))
print()
print()

print("%s seconds" %(time.time()-start_time))
