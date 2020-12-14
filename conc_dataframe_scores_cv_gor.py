import pandas as pd
import os
import argparse
import csv
from math import sqrt
parser=argparse.ArgumentParser()
parser.add_argument("-csv")
parser.add_argument("-o")
args=parser.parse_args()
#path="./profile_dssp_GOR_training/data_frame_scores/"
all_files=os.listdir(args.csv)
#print(all_files)
li=[]
for filename in all_files:
    df=pd.read_csv(args.csv+filename,sep="\t",index_col=None,header=0)
    #print(df)
    li.append(df)
frame=pd.concat(li,axis=1, ignore_index=True)

del frame[3]
del frame[4]
del frame[6]
del frame[7]
del frame[9]
del frame[10]
del frame[12]
del frame[13]
avg=frame.sum(axis=1)/5
frame["avg"]=avg
sd=frame.std(axis=1)
se=sd/sqrt(5)
frame["SD"]=sd
frame["SE"]=se
#print(frame)
#frame=pd.merge([i for i in li], left_index=True,right_index=True,how="outer")
with open(args.o,"w") as output_file:
    output_file.write(frame.to_csv(sep="\t"))
