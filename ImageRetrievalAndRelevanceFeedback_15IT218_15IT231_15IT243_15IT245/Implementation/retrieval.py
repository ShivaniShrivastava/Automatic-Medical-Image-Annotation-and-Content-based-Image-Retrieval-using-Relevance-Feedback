import numpy as np
import pandas as pd

df=pd.read_csv('pred.csv')
data=df.as_matrix()
rows,cols=df.shape
# print (rows)
class_input=input("Enter class number to be retrieved: ")
result=[]
precision=0
tot_rel=0
tot_ret=0
rel_ret=0
for i in range(1,rows):
	print(data[i][1],data[i][3])
	if(int(data[i][3])==int(class_input)):
		tot_ret+=1
	if(int(data[i][1])==int(class_input)):
		tot_rel+=1
	if(int(data[i][3])==int(data[i][1])):
		if(int(data[i][3])==int(class_input)):
			rel_ret+=1
			result.append(data[i][2])
print("Retrieved files are: ",result)
print ("Precision: ",float(rel_ret/(1.0*tot_ret)))
print ("Recall: ",float(rel_ret/(1.0*tot_rel)))
# for i in range(1,rows):
# 	# print (i)
# 	if (data[i][2]==class_input):
# 		if(data[i][1]==data[i][3]):
# 			prec_num=prec_num+1
# 	elif(data[i][2]==class_input):
# 		prec_dec=prec_dec+1
# precision=float(prec_num/(1.0*prec_den))

# for j in range(1,rows):
# 	if(data[j][2]==class_input):
# 		count=count+1
# recall=float(prec_num/(count*1.0))
# print (result,len(result))
