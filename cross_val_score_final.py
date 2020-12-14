from math import sqrt
import os
import argparse
import numpy as np
import pandas as pd
import csv
parser=argparse.ArgumentParser()
parser.add_argument("-obs")
parser.add_argument("-pred")
parser.add_argument("-train_set")
parser.add_argument("-test_set") #folder
parser.add_argument("-o")
args=parser.parse_args()
#id_f=open(args.set)

def binary_matrix(L1,L2,ss):
    bin_mat=np.zeros((2,2), dtype=float)
    #print(bin_mat[1][0])
    for el in range(len(L1)):
        if L1[el]==ss:
            if L2[el]==L1[el]:
                bin_mat[0][0]+=1
            else:
                bin_mat[0][1]+=1
        elif L1[el]!=ss:
            if L2[el]==ss:
                bin_mat[1][0]+=1
            else:
                bin_mat[1][1]+=1

    #print(bin_mat)
    return(bin_mat)

def sensitivity(LI):
    Sen=LI[0][0]/(LI[0][0]+LI[0][1])
    return(Sen)

def PPV(LI):
    ppv=LI[0][0]/(LI[0][0]+LI[1][0])
    return(ppv)

def MCC(LI):
    c=LI[0][0]
    u=LI[0][1]
    o=LI[1][0]
    n=LI[1][1]

    mcc= ((c*n)-(o*u))/(sqrt((c+o)*(c+u)*(n+o)*(n+u)))
    return(mcc)


dssp=os.listdir(args.obs)
pred=os.listdir(args.pred)


L_H=np.zeros((2,2), dtype=float)
L_C=np.zeros((2,2), dtype=float)
L_E=np.zeros((2,2), dtype=float)
N=0
for filename in dssp:
    for filename1 in pred:
        if filename[:-5]==filename1[:-5]:
            f=open(args.obs+filename,"r")
            f1=open(args.pred+filename1,"r")
            for line in f:
                if ">" not in line:
                    line=line.rstrip()
                    L1=list(line[:len(line)])
                    N=N+len(L1)
            for line in f1:
                if ">" not in line:
                    line=line.rstrip()
                    L2=list(line[:len(line)])


            sec=["H","C","E"]
            for ss in sec:
                L=binary_matrix(L1,L2,ss)
                if ss=="H":
                    L_H=L_H+L

                elif ss=="C":
                    L_C=L_C+L

                elif ss=="E":
                     L_E=L_E+L
#print(L_E)
#print(N)
P_HH=L_H[0][0]
print(P_HH)
P_CC=L_C[0][0]
P_EE=L_E[0][0]
ppv_l=[]
mcc_l=[]
index=[]
sen_l=[]
LISTS=[(L_H,"H"),(L_E,"E"),(L_C,"C")]
#print(LISTS)
for el in LISTS:
    LI=el[0]
    sen=sensitivity(LI)
    ppv=PPV(LI)
    mcc=MCC(LI)
    #print(L[1])
    sen_l.append(sen)
    ppv_l.append(ppv)
    mcc_l.append(mcc)
    #print("SEN",sen)
    #print("PPV",ppv)
    #print("MCC",mcc)
    index+=el[1]
    #print(data,index)
Q_l=[]
Q=(P_HH+P_CC+P_EE)/N
Q_l.append(Q)

df_sen=pd.DataFrame(sen_l,index=index)
df_ppv=pd.DataFrame(ppv_l,index=index)
df_mcc=pd.DataFrame(mcc_l,index=index)
ind=["Q"]
df_Q=pd.DataFrame(Q_l, index=ind)
data_conc=pd.concat([df_sen,df_ppv,df_mcc, df_Q],ignore_index=False, keys=["Sen","PPV","MCC","ACC"])
#print(data_conc)
'''with open(args.o,"w") as output_file:
    output_file.write(data_conc.to_csv(sep="\t"))'''
