import argparse
import os
import pandas as pd
import numpy as np
import math
parser=argparse.ArgumentParser()
parser.add_argument("-p") #testing profile as input
#parser.add_argument("-d") #dssp of the  training profiles
parser.add_argument("-m", type=argparse.FileType("r")) #train gor model
parser.add_argument("-train_set")
parser.add_argument("-test_set")
parser.add_argument("-w")
args=parser.parse_args()


def input_prep(profile):  #INPUT: BLINDSET PROFILES
    w_size=int(args.w)//2  #window size
    list_of_rows=[]
    for rows in profile: #iterate over profile lines
        new_rows=rows.rstrip() #edit rows to remove tabs and spaces
        new_rows1=new_rows.split("\t")
        list_of_rows.append(new_rows1[2:])  #PROFILE LIST OF LIST

    list_aa=list_of_rows[0] #LIST OF 20 AA
    list_of_counts=list_of_rows[1:] #LIST OF VALUES
    #print(len(list_of_counts))
    for i in range(w_size):
        list_of_counts.append([0.0 for z in range(20)]) #PROFILE WITH EMPTY LISTS
        list_of_counts=[[0.0 for z in range(20)]]+list_of_counts
    array_counts=np.array([np.array(x) for x in list_of_counts])
    array_counts_float=array_counts.astype(np.float)
    return(array_counts_float,list_aa,w_size)

w=args.w
model=args.m
df=pd.read_csv(model, sep="\t")
matrix_profile=df.values
print(matrix_profile)
def P_SS(matrix_profile): #MARGINAL PROBABILITY OF SS
    P_H=0
    P_C=0
    P_E=0
    for sublist in matrix_profile:
        for value in range(len(sublist)):
            if sublist[1]==0 and sublist[value]=="H": #central residue probability
                P_H=np.sum(sublist[2:]) #MARGINAL PROBABILITY OF SECONDARY STRUCTURE IN CENTRAL POSITIon                #print(sublist)
                continue
            elif sublist[value]=="E" and sublist[1]==0:
                P_E=np.sum(sublist[2:])
                continue
            elif sublist[1]==0 and sublist[value]=="C":
                P_C=np.sum(sublist[2:])
                continue
    return(P_H,P_C,P_E)

PH,PC,PE=P_SS(matrix_profile)
#print(PC)
#print(P_SS(matrix_profile))
matrix_model1=matrix_profile[:len(matrix_profile)-int(w)] #up to len of the matrix -17
matrix_model_r=matrix_profile[len(matrix_profile)-int(w):] #extract from the model the R matrix
matrix_model_H=matrix_model1[:len(matrix_model1)-int(w)*2] #extract from the model the H matrix
matrix_model_C=matrix_model1[len(matrix_model1)-int(w):] #extract from the model the C matrix
matrix_model2=matrix_model1[:len(matrix_model1)-int(w)]
matrix_model_E=matrix_model2[len(matrix_model2)-int(w):]


def information(matrix_model_H,matrix_model_C,matrix_model_E, matrix_model_r): #COMPUTE THE INFORMATION FUNCTION
    matrix_mod_h=np.zeros((int(args.w),22))  #INITIALIZE MATRICES 17X22
    matrix_mod_c=np.zeros((int(args.w),22))
    matrix_mod_e=np.zeros((int(args.w),22))

    wi=list(range(int(args.w))) #list of integers from 0 to 16 (len=17)

    for sublist in wi: #using the integers from 0 to 16 as indices
        for value in range(2,22): #using the integers from 2 to 21 as indices --> compute the information function using the MARGINAL PROBABILITIES AND THE PROBABILITIES OF THE GOR MODEL

            I_H=math.log2(matrix_model_H[sublist][value]/(PH*matrix_model_r[sublist][value])) #INFOFUN=LOG(P(S|R)/P(S)*P(R))
            matrix_mod_h[sublist][value]=I_H #Substitute the values of the info function into the corresponding position of the H model matrix
            matrix_mod_h[sublist][1]=matrix_model_H[sublist][1]
            matrix_mod_h=matrix_mod_h.astype("object")
            matrix_mod_h[sublist][0]="H"


            I_C=math.log2(matrix_model_C[sublist][value]/(PC*matrix_model_r[sublist][value]))
            matrix_mod_c[sublist][value]=I_C
            matrix_mod_c[sublist][1]=matrix_model_C[sublist][1]
            matrix_mod_c=matrix_mod_c.astype("object")
            matrix_mod_c[sublist][0]="C"


            I_E=math.log2(matrix_model_E[sublist][value]/(PE*matrix_model_r[sublist][value]))
            matrix_mod_e[sublist][value]=I_E
            matrix_mod_e[sublist][1]=matrix_model_E[sublist][1]
            matrix_mod_e=matrix_mod_e.astype("object")
            matrix_mod_e[sublist][0]="E"
    #print(matrix_mod_e)
    #print(matrix_mod_c)
    #print(matrix_mod_h)
    return(matrix_mod_h,matrix_mod_c,matrix_mod_e)

information(matrix_model_H,matrix_model_C,matrix_model_E, matrix_model_r)
I_model_H,I_model_C,I_model_E=information(matrix_model_H,matrix_model_C,matrix_model_E, matrix_model_r)
#print(I_model_H)

I_model_C_n=[]
I_model_H_n=[]
I_model_E_n=[]
for sublist in range(len(I_model_H)): #append ONLY THE VALUES IN NEW LISTS (REDUNDANT!!!)
    I_model_H_n.append(I_model_H[sublist][2:])
    I_model_C_n.append(I_model_C[sublist][2:])
    I_model_E_n.append(I_model_E[sublist][2:])


testing_profile=os.listdir(args.p)
set_path="./cv/"
file_set=open(set_path+"set"+args.test_set)
for id in file_set:
    id=id.rstrip("\n")
    if id+".fasta.pssm.profile" in testing_profile:
        ss=""
        profile=open(args.p+id+".fasta.pssm.profile","r")
        profile_array, list_aa,w_size=input_prep(profile)
        #print(type(profile_array[20][8]))
        for i in range(w_size,len(profile_array)-w_size): #SLIDING WINDOW
            window=[]
            window=profile_array[i-w_size:i+w_size+1]
            LH=[] #initialize new lists to append the results of the multiplication between the info fun and the prob of the profile
            LC=[]
            LE=[]
            for sub in range(len(I_model_H_n)):
                for el in range(len(window)):
                    h_result=I_model_H_n[sub][el]*window[sub][el]
                    LH.append(h_result)
                    c_result=I_model_C_n[sub][el]*window[sub][el]
                    LC.append(c_result)
                    e_result=I_model_E_n[sub][el]*window[sub][el]
                    LE.append(e_result)
            sum_h=np.sum(LH) #compute the sum of all the values for H C AND E--> overall probability of the ss
            #print("h",sum_h, "LH", LH)
            sum_c=np.sum(LC)
            #print("c",sum_c)
            sum_e=np.sum(LE)
            #print("e",sum_e)
            probH=(sum_h, "H")
            probC=(sum_c, "C")
            probE=(sum_e, "E")
            #print(probC)
            if probH[0]==0 and probC[0]==0 and probE[0]==0:
                ss+="C"
            else:
                MAX_S=max(probH,probC,probE) #compute the max probabilities among the available three probabilities
                ss+=MAX_S[1]
                #print(ss)

        filename_out=id+".pred"
        path="./cv_set"+args.train_set+"/"
        '''with open(os.path.join("./cv_set"+args.train_set+"/"+filename_out),"w") as output_file:
            output_file.write(">"+filename_out+"\n")
            output_file.write(ss)'''
