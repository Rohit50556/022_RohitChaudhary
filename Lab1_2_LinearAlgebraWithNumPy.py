import numpy as np

##1+2
arr1=np.random.rand(3,2)*10+1
arr2=np.random.rand(2,3)*10+1

##3
print(np.dot(arr1,arr2))

##4
res = [[0 for x in range(len(arr1))] for y in range(len(arr2[0]))]

for i in range(len(arr1)): 
    for j in range(len(arr2[0])): 
        for k in range(len(arr2)): 
            res[i][j] += arr1[i][k] * arr2[k][j]

print (res)

##5
res=np.mean(arr1)
print(res)

