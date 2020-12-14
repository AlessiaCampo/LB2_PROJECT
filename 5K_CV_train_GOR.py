import argparse
import os
import numpy as np
import pandas as pd
parser=argparse.ArgumentParser()
parser.add_argument("-p")
parser.add_argument("-d")
parser.add_argument("-s")
parser.add_argument("-w")
parser.add_argument("-o")
args=parser.parse_args()
sets=args.s



def input_prep(dssp, profile):  #dssp= dssp files, profile=profiles for training set
    w_size=int(args.w)//2
    list_of_rows=[]
    for line in dssp: #iterate over dssp files and put each row of the dssp sequence into a list
        if ">" not in line:
            line=line.rstrip()
            s=list("".join(line))
            ss=w_size*[""]+s+[""]*w_size
    for rows in profile: #iterate over profiles of training set
        new_rows=rows.rstrip()
        new_rows1=new_rows.split("\t") #edit to remove tabs and spaces
        list_of_rows.append(new_rows1[2:])

    list_aa=list_of_rows[0] #save on a new list the list of the 20 AA
    list_of_counts=list_of_rows[1:] # SAVE ON ANOTHER LIST THE VALUES (PROFILE PROBABILITIES)

    for i in range(w_size): #PROFILE WITH EMPTY LISTS--> need to add list of zeros as many as the window size
        list_of_counts.append([0.0 for z in range(20)])  #at the end of the profile
        list_of_counts=list_of_counts=[[0.0 for z in range(20)]]+list_of_counts #at the beginning

    array_counts=np.array([np.array(x) for x in list_of_counts]) #convert each value of the profile into a float number
    array_counts_float=array_counts.astype(np.float)
    return(array_counts_float, ss, w_size,list_aa)

w=args.w
def matrices(w):  # 0 MATRICES INITIALIZED for R(RESIDUES), H(HELIX),E(STRANDS), C(COIL)
    R=np.zeros((int(w),20), dtype=float)  #the size should be 17x20 (where 17 is the window size)
    H=np.zeros((int(w),20), dtype=float)
    E=np.zeros((int(w),20), dtype=float)
    C=np.zeros((int(w),20), dtype=float)

    return(R,H,E,C)

R,H,E,C=matrices(w)



#MAIN PROGRAM
dssp_l=os.listdir(args.d)
profile_l=os.listdir(args.p)
#print(profile_l)
set_path="./cv/"
for num in sets:
    file_set=open(set_path+"set"+num)
    for id in file_set:
        id=id.rstrip("\n")
        dssp=open(args.d +id+".dssp","r")
        if id+".fasta.pssm.profile" in profile_l:
            profile=open(args.p +id+ ".fasta.pssm.profile",  "r")
            list_of_counts, ss,w_size,list_aa= input_prep(dssp,profile)
            for i in range(w_size,len(ss)-w_size): #8,len(ss-8) SLIDING WINDOW
                w_list=[]
                w_list=list_of_counts[i-w_size:i+w_size+1] #add to the sliding windows the correspondent values in the lists of counts (profile) slinding from i-8 to i+8 at each iteration

                for el in range(len(w_list)): #iterate over each single sliding window
                    R[el]+=w_list[el] #fill the R table with the values in the current sliding window
                    if ss[i]=="H": #fill the H table respect to the secondary structure in the i position of the sec structure
                        H[el]+=w_list[el]
                    if ss[i]=="E": #fill the E table
                        E[el]+=w_list[el]
                    if ss[i]=="C": #fill the C table
                        C[el]+=w_list[el]

def normalize(H,E,C,R): #NORMALIZE DATA
    r_sum=R.sum(axis=1, dtype="float") #vector
    norm_h=H/r_sum.reshape((int(w),1))
    norm_c=C/r_sum.reshape((int(w),1))
    norm_e=E/r_sum.reshape((int(w),1))
    norm_r=R/r_sum.reshape((int(w),1))

    return(norm_h,norm_c,norm_e, norm_r)

normalize(H,E,C,R)
H_table,C_table, E_table,R_table=normalize(H,E,C,R)

index=[i for i in range(-w_size,w_size+1)]
E_df=pd.DataFrame(E_table,index=index)
C_df=pd.DataFrame(C_table, index=index)
H_df=pd.DataFrame(H_table, index=index)
R_df=pd.DataFrame(R_table, index=index)
data_frames=pd.concat([H_df,E_df, C_df, R_df], ignore_index=False, keys=["H","E","C","R"])
data_frames.columns=[list_aa]

path="./cv_set"+args.s+"/"
#print(path)
'''with open(os.path.join("./cv_set"+args.s+"/"+args.o), "w") as output_file:

    output_file.write(data_frames.to_csv(sep="\t"))'''
